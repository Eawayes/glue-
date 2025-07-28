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
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class RTEDataset(torch.utils.data.Dataset):
    """RTE数据集加载器"""

    def __init__(self, data_dir, split='train', max_length=128, model_name='bert-base-uncased'):
        """
        初始化RTE数据集
        :param data_dir: 数据集根目录
        :param split: 数据分割 (train, dev, test)
        :param max_length: 最大序列长度
        :param model_name: 预训练模型名称
        """
        self.split = split
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 构建文件路径
        file_path = os.path.join(data_dir, "RTE", f"{split}.tsv")

        try:
            df = pd.read_csv(file_path, sep='\t')
            # 检查必要的列是否存在
            if 'sentence1' in df.columns and 'sentence2' in df.columns and 'label' in df.columns:
                df = df[['sentence1', 'sentence2', 'label']]
                # 标签映射: entailment->1, not_entailment->0
                df['label'] = df['label'].astype(str).apply(
                    lambda x: 1 if x.lower() == "entailment" else 0
                )
            else:
                print(f"Warning: Required columns missing in RTE {split} set. Using empty DataFrame.")
                df = pd.DataFrame(columns=['sentence1', 'sentence2', 'label'])
        except Exception as e:
            print(f"Error loading RTE {split} set: {e}")
            df = pd.DataFrame(columns=['sentence1', 'sentence2', 'label'])

        self.data = df.reset_index(drop=True)
        print(f"Loaded {len(self.data)} samples for RTE {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text1 = str(row['sentence1'])
        text2 = str(row['sentence2'])
        label = torch.tensor(row['label'], dtype=torch.long)

        encoding = self.tokenizer(
            text1, text2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }


def train_rte_model(data_dir, model_name='bert-base-uncased',
                    epochs=10, batch_size=16, lr=1e-5,
                    max_length=256, patience=3):
    """训练并评估RTE模型，包含早停机制"""
    print(f"\n{'=' * 50}")
    print(f"Training RTE model with enhanced parameters")
    print(f"Model: {model_name}, Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Learning rate: {lr}, Max length: {max_length}")
    print(f"{'=' * 50}")

    # 创建数据集
    try:
        train_dataset = RTEDataset(data_dir, 'train', max_length, model_name)
        dev_dataset = RTEDataset(data_dir, 'dev', max_length, model_name)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return {}, None, None

    if len(train_dataset) == 0 or len(dev_dataset) == 0:
        print("Skipping RTE due to empty dataset")
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
    use_amp = device.type == 'cuda'
    if use_amp:
        scaler = GradScaler()
        print("Using mixed precision training")
    else:
        print("Disabling mixed precision")

    # 初始化模型
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}, None, None

    # 保存tokenizer用于预测
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 设置优化器 - 添加权重衰减
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # 训练循环变量
    best_accuracy = 0.0
    epochs_without_improvement = 0
    training_stats = []

    # 训练循环
    for epoch in range(epochs):
        print(f"\n{'=' * 20} Epoch {epoch + 1}/{epochs} {'=' * 20}")
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training")

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

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

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf loss detected. Skipping batch.")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**inputs)
                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf loss detected. Skipping batch.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

            # 更新进度条
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        # 计算平均训练损失
        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # 验证
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Validating"):
                inputs = {
                    'input_ids': batch['input_ids'].to(device, non_blocking=True),
                    'attention_mask': batch['attention_mask'].to(device, non_blocking=True)
                }
                labels = batch['labels'].cpu().numpy()

                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        # 计算验证准确率
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Validation Accuracy: {accuracy:.4f}")

        # 记录训练统计
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_accuracy': accuracy
        }
        training_stats.append(epoch_stats)

        # 早停检查
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
            # 保存最佳模型
            best_model_state = model.state_dict()
            print(f"New best accuracy: {best_accuracy:.4f}. Saving model.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs")

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # 加载最佳模型
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with accuracy: {best_accuracy:.4f}")

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
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # 计算最终准确率
    final_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Final Validation Accuracy: {final_accuracy:.4f}")

    metrics = {
        "best_accuracy": best_accuracy,
        "final_accuracy": final_accuracy,
        "training_stats": training_stats
    }

    return metrics, model, tokenizer


def generate_rte_submission(model, tokenizer, data_dir, max_length=256):
    """生成RTE测试集预测文件"""
    print("\nGenerating RTE submission file...")

    # 创建测试数据集
    try:
        dataset = RTEDataset(data_dir, 'test', max_length)
    except Exception as e:
        print(f"Error creating test dataset: {e}")
        return None

    if len(dataset) == 0:
        print("Skipping submission due to empty test dataset")
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
    all_ids = []

    try:
        # 读取测试集ID
        test_file = os.path.join(data_dir, "RTE", "test.tsv")
        test_df = pd.read_csv(test_file, sep='\t')
        if 'index' in test_df.columns:
            all_ids = test_df['index'].tolist()
        else:
            # 如果没有ID列，创建顺序ID
            all_ids = list(range(len(dataset)))

        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                inputs = {
                    'input_ids': batch['input_ids'].to(device, non_blocking=True),
                    'attention_mask': batch['attention_mask'].to(device, non_blocking=True)
                }

                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy().astype(int)
                all_preds.extend(preds)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

    # 创建提交目录
    os.makedirs("rte_submission", exist_ok=True)
    submission_file = os.path.join("rte_submission", "RTE.tsv")

    # 保存为GLUE要求的格式
    try:
        with open(submission_file, 'w', encoding='utf-8') as f:
            f.write("index\tprediction\n")
            for idx, pred in zip(all_ids, all_preds):
                f.write(f"{idx}\t{pred}\n")
    except Exception as e:
        print(f"Error saving submission: {e}")
        return None

    print(f"Saved {len(all_preds)} predictions to {submission_file}")
    return submission_file


def plot_training_stats(stats):
    """绘制训练统计图表"""
    import matplotlib.pyplot as plt

    epochs = [s['epoch'] for s in stats]
    train_losses = [s['train_loss'] for s in stats]
    val_accuracies = [s['val_accuracy'] for s in stats]

    plt.figure(figsize=(12, 5))

    # 训练损失图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 验证准确率图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'r-o', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('rte_training_stats.png')
    plt.show()


if __name__ == "__main__":
    # 配置参数 - 针对RTE任务优化
    DATA_DIR = "./glue"  # GLUE数据集路径
    MODEL_NAME = "bert-base-uncased"  # 基础模型

    # RTE优化参数
    EPOCHS = 15  # 增加训练轮数
    BATCH_SIZE = 8  # 减小批大小以增加更新次数
    LEARNING_RATE = 2e-5  # 微调学习率
    MAX_LENGTH = 256  # 增加序列长度以捕获更多上下文
    PATIENCE = 4  # 早停耐心值

    print("\nStarting RTE model training with optimized parameters...")

    # 训练RTE模型
    metrics, model, tokenizer = train_rte_model(
        data_dir=DATA_DIR,
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        max_length=MAX_LENGTH,
        patience=PATIENCE
    )

    # 打印最终结果
    if metrics:
        print("\n" + "=" * 50)
        print("RTE Training Results:")
        print(f"Best Validation Accuracy: {metrics['best_accuracy']:.4f}")
        print(f"Final Validation Accuracy: {metrics['final_accuracy']:.4f}")
        print("=" * 50)

        # 绘制训练统计
        plot_training_stats(metrics['training_stats'])

    # 生成提交文件
    if model and tokenizer:
        submission_file = generate_rte_submission(
            model=model,
            tokenizer=tokenizer,
            data_dir=DATA_DIR,
            max_length=MAX_LENGTH
        )
        if submission_file:
            print(f"\nGLUE submission file created: {submission_file}")

    # 保存模型
    if model:
        model_path = "rte_optimized_model"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"\nSaved optimized model to {model_path}")

    print("\nRTE Optimization Complete!")