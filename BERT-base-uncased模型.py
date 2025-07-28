import os

# 设置Hugging Face镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 禁用符号链接警告
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# 禁用不必要的日志
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class GLUEDataset(torch.utils.data.Dataset):
    """自定义GLUE数据集加载器"""

    def __init__(self, task_name, data_dir, split='train', max_length=128, model_name='bert-base-uncased'):
        """
        初始化GLUE数据集
        :param task_name: 任务名称 (SST-2, RTE, STS-B)
        :param data_dir: 数据集根目录
        :param split: 数据分割 (train, dev, test)
        :param max_length: 最大序列长度
        :param model_name: 预训练模型名称
        """
        self.task_name = task_name
        self.split = split
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 构建文件路径
        task_dir = os.path.join(data_dir, task_name)
        file_path = os.path.join(task_dir, f"{split}.tsv")

        # 读取并处理不同格式的数据集
        try:
            if task_name == "SST-2":
                # SST-2数据集格式: 文本, 标签
                df = pd.read_csv(file_path, sep='\t', header=0, names=['text', 'label'])
            elif task_name == "RTE":
                # RTE数据集格式: 句子1, 句子2, 标签
                df = pd.read_csv(file_path, sep='\t')
                # 检查必要的列是否存在
                if 'sentence1' in df.columns and 'sentence2' in df.columns and 'label' in df.columns:
                    df = df[['sentence1', 'sentence2', 'label']]
                else:
                    print(f"Warning: Required columns missing in RTE {split} set. Using empty DataFrame.")
                    df = pd.DataFrame(columns=['sentence1', 'sentence2', 'label'])
            elif task_name == "STS-B":
                # STS-B数据集格式: 尝试多种格式
                try:
                    # 格式1：有标题行
                    df = pd.read_csv(file_path, sep='\t', header=0)
                    if 'sentence1' in df.columns and 'sentence2' in df.columns and 'score' in df.columns:
                        df = df[['sentence1', 'sentence2', 'score']]
                    elif 'text1' in df.columns and 'text2' in df.columns and 'score' in df.columns:
                        df = df[['text1', 'text2', 'score']]
                    else:
                        # 格式2：无标题行
                        df = pd.read_csv(file_path, sep='\t', header=None,
                                         names=['id', 'score', 'text1', 'text2'])
                        df = df[['text1', 'text2', 'score']]
                except Exception as e:
                    print(f"Error loading STS-B data: {e}")
                    # 格式3：GLUE官方格式
                    try:
                        df = pd.read_csv(file_path, sep='\t', header=0, usecols=[7, 8, 9],
                                         names=['text1', 'text2', 'score'])
                    except:
                        df = pd.DataFrame()

                if not df.empty:
                    df['score'] = pd.to_numeric(df['score'], errors='coerce')
                    df = df.dropna()
        except Exception as e:
            print(f"Error loading {task_name} {split} set: {e}")
            df = pd.DataFrame()

        # 处理标签映射和类型
        if not df.empty and task_name != "STS-B":
            # RTE标签映射
            if task_name == "RTE":
                # 先将标签转换为字符串，然后映射
                df['label'] = df['label'].astype(str).apply(lambda x: 1 if x.lower() == "entailment" else 0)
            else:
                # 确保标签是数值类型
                if 'label' in df.columns:
                    df['label'] = pd.to_numeric(df['label'], errors='coerce')
                    df['label'] = df['label'].fillna(0).astype(int)
                else:
                    # 如果没有标签列，创建一列全0
                    df['label'] = 0

        # 对于回归任务
        if task_name == "STS-B":
            if 'score' in df.columns:
                df['score'] = pd.to_numeric(df['score'], errors='coerce')
                df['score'] = df['score'].fillna(0.0)
            else:
                df['score'] = 0.0

        # 对于测试集，确保有必要的列
        if split == 'test':
            if task_name == "SST-2" and 'text' not in df.columns:
                if 'sentence' in df.columns:
                    df = df.rename(columns={'sentence': 'text'})
                elif 'sentence1' in df.columns:
                    df = df.rename(columns={'sentence1': 'text'})

            if task_name in ["RTE", "STS-B"]:
                if 'sentence1' not in df.columns and 'text1' in df.columns:
                    df = df.rename(columns={'text1': 'sentence1'})
                if 'sentence2' not in df.columns and 'text2' in df.columns:
                    df = df.rename(columns={'text2': 'sentence2'})

        self.data = df.reset_index(drop=True)

        if self.data.empty:
            print(f"Warning: Empty dataset for {task_name} {split}")
        else:
            print(f"Loaded {len(self.data)} samples for {task_name} {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 分类任务处理
        if self.task_name == "SST-2":
            text = str(row['text']) if 'text' in row else ""
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            label = torch.tensor(row['label'], dtype=torch.long)

        # 句对任务处理
        elif self.task_name == "RTE":
            text1 = str(row['sentence1']) if 'sentence1' in row else ""
            text2 = str(row['sentence2']) if 'sentence2' in row else ""
            encoding = self.tokenizer(
                text1, text2,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            label = torch.tensor(row['label'], dtype=torch.long)

        # 回归任务处理 (STS-B)
        elif self.task_name == "STS-B":
            text1 = str(row['text1']) if 'text1' in row else row.get('sentence1', "")
            text2 = str(row['text2']) if 'text2' in row else row.get('sentence2', "")
            score = row['score']
            encoding = self.tokenizer(
                text1, text2,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            label = torch.tensor(score, dtype=torch.float)
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")

        # 返回统一格式
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }


def train_model(task_name, data_dir, model_name='bert-base-uncased',
                epochs=3, batch_size=32, lr=2e-5, max_length=128):
    """训练并评估单个GLUE任务模型"""
    print(f"\n{'=' * 50}")
    print(f"Training model for task: {task_name}")
    print(f"{'=' * 50}")

    # 创建数据集
    try:
        train_dataset = GLUEDataset(task_name, data_dir, 'train', max_length, model_name)
        dev_dataset = GLUEDataset(task_name, data_dir, 'dev', max_length, model_name)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return {}, None, None

    if len(train_dataset) == 0 or len(dev_dataset) == 0:
        print(f"Skipping {task_name} due to empty dataset")
        return {}, None, None

    # 创建数据加载器
    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 决定是否使用混合精度
    use_amp = len(train_loader) > 50 and device.type == 'cuda'
    if use_amp:
        scaler = GradScaler()
        print("Using mixed precision training")
    else:
        print("Disabling mixed precision")

    # 初始化模型 (STS-B是回归任务，其他是分类)
    num_labels = 1 if task_name == "STS-B" else 2
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(device)
    except Exception as e:
        print(f"Error loading model for {task_name}: {e}")
        return {}, None, None

    # 保存tokenizer用于预测
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 设置优化器 - 添加权重衰减
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 添加warmup阶段
        num_training_steps=total_steps
    )

    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        valid_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            # 准备输入数据 - 直接移动到GPU
            inputs = {
                'input_ids': batch['input_ids'].to(device, non_blocking=True),
                'attention_mask': batch['attention_mask'].to(device, non_blocking=True),
                'labels': batch['labels'].to(device, non_blocking=True)
            }

            # 混合精度训练
            if use_amp:
                with autocast():
                    outputs = model(**inputs)
                    loss = outputs.loss

                # 检查损失是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf loss detected at batch {batch_idx}. Skipping batch.")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**inputs)
                loss = outputs.loss

                # 检查损失是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf loss detected at batch {batch_idx}. Skipping batch.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            valid_batches += 1

            # 每100个批次显示一次进度
            if batch_idx % 100 == 0:
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}"
                )

        # 计算平均损失
        if valid_batches > 0:
            avg_train_loss = total_loss / valid_batches
            print(f"Average training loss: {avg_train_loss:.4f}")
        else:
            print("No valid batches in this epoch")
            continue

        # 每轮结束后进行验证
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Validating Epoch {epoch + 1}"):
                inputs = {
                    'input_ids': batch['input_ids'].to(device, non_blocking=True),
                    'attention_mask': batch['attention_mask'].to(device, non_blocking=True)
                }
                labels = batch['labels'].cpu().numpy()

                outputs = model(**inputs)
                logits = outputs.logits

                if task_name == "STS-B":
                    preds = logits.squeeze().cpu().numpy()
                else:
                    preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        # 计算并打印验证指标
        if len(all_labels) > 0:
            if task_name == "STS-B":
                # 计算Pearson相关系数
                pearson_corr = pearsonr(all_labels, all_preds)[0]
                print(f"Validation Pearson Correlation: {pearson_corr:.4f}")
            else:
                # 计算准确率
                accuracy = accuracy_score(all_labels, all_preds)
                print(f"Validation Accuracy: {accuracy:.4f}")

        model.train()

    # 最终评估
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Final Evaluation"):
            inputs = {
                'input_ids': batch['input_ids'].to(device, non_blocking=True),
                'attention_mask': batch['attention_mask'].to(device, non_blocking=True)
            }
            labels = batch['labels'].cpu().numpy()

            outputs = model(**inputs)
            logits = outputs.logits

            # 处理预测结果
            if task_name == "STS-B":  # 回归任务
                preds = logits.squeeze().cpu().numpy()
            else:  # 分类任务
                preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # 计算评估指标
    metrics = {}

    if len(all_labels) == 0:
        print(f"No data to evaluate for {task_name}")
    else:
        if task_name == "STS-B":
            # Pearson相关系数计算
            pearson_corr = pearsonr(all_labels, all_preds)[0]
            metrics["pearson"] = pearson_corr
            print(f"Final Pearson Correlation: {pearson_corr:.4f}")
        else:
            # 准确率计算
            accuracy = accuracy_score(all_labels, all_preds)
            metrics["accuracy"] = accuracy
            print(f"Final Accuracy: {accuracy:.4f}")

    return metrics, model, tokenizer


def generate_glue_submission(task_name, model, tokenizer, data_dir, max_length=128):
    """
    生成符合GLUE提交格式的测试集预测文件
    文件命名格式: SST-2.tsv, RTE.tsv, STS-B.tsv
    """
    # 创建测试数据集
    try:
        dataset = GLUEDataset(task_name, data_dir, 'test', max_length)
    except Exception as e:
        print(f"Error creating test dataset for {task_name}: {e}")
        return None

    if len(dataset) == 0:
        print(f"Skipping submission for {task_name} due to empty test dataset")
        return None

    # 创建数据加载器
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )

    # 生成预测
    model.eval()
    all_preds = []

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Generating test predictions for {task_name}"):
                inputs = {
                    'input_ids': batch['input_ids'].to(device, non_blocking=True),
                    'attention_mask': batch['attention_mask'].to(device, non_blocking=True)
                }

                outputs = model(**inputs)
                logits = outputs.logits

                if task_name == "STS-B":  # 回归任务
                    preds = logits.squeeze().cpu().numpy()
                    # 确保预测值在 [0, 5] 范围内
                    preds = np.clip(preds, 0, 5)
                else:  # 分类任务
                    preds = torch.argmax(logits, dim=1).cpu().numpy().astype(int)

                all_preds.extend(preds)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

    # 创建提交目录
    os.makedirs("glue_submissions", exist_ok=True)

    # 保存为GLUE要求的格式 (使用任务名作为文件名)
    submission_file = os.path.join("glue_submissions", f"{task_name}.tsv")

    try:
        with open(submission_file, 'w', encoding='utf-8') as f:
            if task_name == "STS-B":
                for pred in all_preds:
                    f.write(f"{pred:.4f}\n")  # 浮点数保留4位小数
            else:
                for pred in all_preds:
                    f.write(f"{int(pred)}\n")  # 整数格式
    except Exception as e:
        print(f"Error saving submission for {task_name}: {e}")
        return None

    print(f"Saved {len(all_preds)} predictions to {submission_file}")
    return submission_file


def compare_glue_performance(data_dir, model_name='bert-base-uncased',
                             epochs=3, batch_size=32, max_length=128):
    """训练并比较GLUE任务的性能（SST-2, RTE, STS-B）"""
    tasks = ["SST-2", "RTE", "STS-B"]

    results = {}
    models = {}
    tokenizers = {}
    training_times = {}

    # 任务特定超参数
    task_lr = {
        "RTE": 1e-5,
        "STS-B": 1e-5,
        "default": 2e-5
    }

    # 任务特定epochs
    task_epochs = {
        "SST-2": 2,
        "RTE": 5,
        "default": epochs
    }

    # 任务特定批处理大小
    task_batch_size = {
        "STS-B": 48,
        "default": 32
    }

    print(f"Training on GLUE tasks: {', '.join(tasks)}")

    for task in tasks:
        print(f"\nStarting task: {task}")
        start_time = time.time()
        metrics = {}
        model = None
        tokenizer = None

        try:
            # 获取任务特定参数
            lr = task_lr.get(task, task_lr["default"])
            bs = task_batch_size.get(task, task_batch_size["default"])
            ep = task_epochs.get(task, task_epochs["default"])

            print(f"Using parameters: lr={lr:.1e}, batch_size={bs}, epochs={ep}")

            # 训练并评估模型
            metrics, model, tokenizer = train_model(
                task, data_dir, model_name, ep, bs, lr, max_length
            )
        except Exception as e:
            print(f"Error training model for {task}: {e}")

        # 记录结果
        end_time = time.time()
        results[task] = metrics
        models[task] = model
        tokenizers[task] = tokenizer
        training_times[task] = end_time - start_time if model else 0

        # 生成GLUE提交文件
        if model and tokenizer:
            try:
                submission_file = generate_glue_submission(
                    task_name=task,
                    model=model,
                    tokenizer=tokenizer,
                    data_dir=data_dir,
                    max_length=max_length
                )
                if submission_file:
                    print(f"GLUE submission file created for {task}: {submission_file}")
            except Exception as e:
                print(f"Error generating GLUE submission for {task}: {e}")

    # 创建结果DataFrame
    result_df = pd.DataFrame.from_dict(results, orient='index')

    # 添加训练时间
    result_df['training_time'] = pd.Series(training_times)

    # 打印结果
    print("\n" + "=" * 50)
    print("GLUE Benchmark Performance Comparison")
    print("=" * 50)
    print(result_df)

    # 可视化结果
    if not result_df.empty:
        visualize_results(result_df)

    return result_df, models


def visualize_results(result_df):
    """可视化GLUE任务性能比较结果"""
    # 筛选出有数据的指标列
    metric_cols = [col for col in result_df.columns if col != 'training_time' and not result_df[col].isnull().all()]

    if not metric_cols:
        print("No metrics to visualize")
        return

    # 创建图表
    plt.figure(figsize=(15, 10))

    # 创建子图布局
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 指标性能条形图
    ax1 = axes[0]
    result_df[metric_cols].plot(kind='bar', ax=ax1)
    ax1.set_title('GLUE Tasks Performance Metrics')
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Task')
    ax1.legend(title='Metrics', loc='best')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_xticklabels(result_df.index, rotation=45)

    # 训练时间条形图
    ax2 = axes[1]
    result_df['training_time'].plot(kind='bar', color='skyblue', ax=ax2)
    ax2.set_title('Training Time per Task (seconds)')
    ax2.set_ylabel('Time (s)')
    ax2.set_xlabel('Task')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_xticklabels(result_df.index, rotation=45)

    plt.tight_layout()
    plt.savefig('glue_performance_comparison.png')
    plt.show()


if __name__ == "__main__":
    # 配置参数
    DATA_DIR = "./glue"  # GLUE数据集路径
    MODEL_NAME = "bert-base-uncased"  # 使用的基础模型
    EPOCHS = 3  # 默认训练轮数
    BATCH_SIZE = 32  # 默认批量大小
    MAX_LENGTH = 128  # 序列最大长度

    # 创建结果目录
    os.makedirs("glue_submissions", exist_ok=True)

    # 训练并比较所有GLUE任务
    results, models = compare_glue_performance(
        data_dir=DATA_DIR,
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH
    )

    # 保存结果
    results.to_csv('glue_performance_results.csv')
    print("\nPerformance results saved to 'glue_performance_results.csv'")
    print("Visualization saved to 'glue_performance_comparison.png'")
    print("GLUE submission files saved to 'glue_submissions' directory")

    # 保存模型
    for task, model in models.items():
        if model is not None:
            model_path = f"{task}_model"
            model.save_pretrained(model_path)
            print(f"Saved model for {task} to {model_path}")

    print("\nGLUE Submission Instructions:")
    print("1. Zip the 'glue_submissions' directory")
    print("2. Visit the GLUE benchmark submission page: https://gluebenchmark.com/submit")
    print("3. Upload the zip file for evaluation")