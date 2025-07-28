import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.pyplot as plt  # 导入绘图库
import matplotlib.font_manager as fm  # 字体管理

# 环境配置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 检查是否支持混合精度训练
if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler

    use_amp = True
    print("Using mixed precision training")
else:
    use_amp = False
    print("Mixed precision not available, using standard training")


# 创建带重试机制的会话
def create_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


# 安全下载模型
def safe_download_model(model_name, model_class=AutoModel):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Downloading {model_name}")
            model = model_class.from_pretrained(model_name)
            print(f"Successfully downloaded {model_name}")
            return model
        except (requests.exceptions.RequestException, OSError) as e:
            print(f"Download error: {e}")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {model_name} after {max_retries} attempts")
                if "large" in model_name:
                    base_model_name = model_name.replace("large", "base")
                    print(f"Trying smaller model: {base_model_name}")
                    return safe_download_model(base_model_name, model_class)
                else:
                    raise
    return None


# 安全下载tokenizer
def safe_download_tokenizer(model_name):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Downloading tokenizer for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"Successfully downloaded tokenizer for {model_name}")
            return tokenizer
        except (requests.exceptions.RequestException, OSError) as e:
            print(f"Download error: {e}")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download tokenizer for {model_name} after {max_retries} attempts")
                if "large" in model_name:
                    base_model_name = model_name.replace("large", "base")
                    print(f"Trying smaller model tokenizer: {base_model_name}")
                    return safe_download_tokenizer(base_model_name)
                else:
                    raise
    return None


class RTEModel(nn.Module):
    """专门为RTE任务优化的模型架构"""

    def __init__(self, model_name="roberta-base", dropout_rate=0.2):
        super(RTEModel, self).__init__()
        try:
            self.bert = safe_download_model(model_name, AutoModel)
        except Exception as e:
            print(f"Critical error loading model: {e}")
            print("Using fallback model: roberta-base")
            self.bert = AutoModel.from_pretrained("roberta-base")

        hidden_size = self.bert.config.hidden_size

        # 增强的分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 2)  # 二分类输出
        )

        # 初始化分类头权重
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 获取不同层的表示
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        cls_token = last_hidden[:, 0, :]

        # 使用平均池化获取序列表示
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # 拼接多种表示
        combined = torch.cat((
            pooler_output,
            cls_token,
            mean_pooled,
            last_hidden[:, -1, :]  # 最后一个token
        ), dim=1)

        logits = self.classifier(combined)
        return logits


class RTEDataset(Dataset):
    """专门为RTE任务优化的数据集加载器"""

    def __init__(self, data_dir, split='train', max_length=192, model_name='roberta-base'):
        self.split = split
        self.max_length = max_length

        try:
            self.tokenizer = safe_download_tokenizer(model_name)
        except Exception as e:
            print(f"Critical error loading tokenizer: {e}")
            print("Using fallback tokenizer: roberta-base")
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # 加载数据
        file_path = os.path.join(data_dir, "RTE", f"{split}.tsv")
        try:
            df = pd.read_csv(file_path, sep='\t')

            # 确保必要的列存在
            required_cols = ['sentence1', 'sentence2', 'label']
            if not all(col in df.columns for col in required_cols):
                if 'text_a' in df.columns and 'text_b' in df.columns:
                    df = df.rename(columns={'text_a': 'sentence1', 'text_b': 'sentence2'})
                elif 'premise' in df.columns and 'hypothesis' in df.columns:
                    df = df.rename(columns={'premise': 'sentence1', 'hypothesis': 'sentence2'})
                else:
                    raise ValueError(f"Required columns missing in RTE {split} set")

            # 处理标签
            df['label'] = df['label'].astype(str).apply(
                lambda x: 1 if x.lower() in ["entailment", "1"] else 0
            )

            # 过滤掉空字符串
            df = df[
                (df['sentence1'].str.strip() != '') &
                (df['sentence2'].str.strip() != '')
                ]

            self.data = df[['sentence1', 'sentence2', 'label']].dropna()
            print(f"Loaded {len(self.data)} samples for RTE {split}")
        except Exception as e:
            print(f"Error loading RTE data: {e}")
            self.data = pd.DataFrame(columns=['sentence1', 'sentence2', 'label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text1 = str(row['sentence1'])
        text2 = str(row['sentence2'])

        # 使用预训练模型的特定格式
        encoding = self.tokenizer(
            text1, text2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        label = torch.tensor(row['label'], dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }


def plot_training_curves(history, model_name="roberta-base"):
    """绘制训练损失和验证准确率曲线（使用英文标签）"""
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(12, 10))

    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    plt.title('Train Loss Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # 绘制准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='green', linewidth=2)
    plt.title('Accuracy Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()

    # 保存图片
    plot_path = os.path.join("plots", f"{model_name}_training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to: {plot_path}")
    return plot_path


def train_rte_model(data_dir, model_name="roberta-base",
                    epochs=10, batch_size=8, lr=1e-5, max_length=192):
    """训练并评估RTE模型"""
    print("\n" + "=" * 50)
    print(f"Training optimized model for RTE task")
    print(f"Using model: {model_name}")
    print("=" * 50)

    # 记录训练开始时间（新增代码）
    start_time = time.time()

    # 创建数据集
    try:
        train_dataset = RTEDataset(data_dir, 'train', max_length, model_name)
        dev_dataset = RTEDataset(data_dir, 'dev', max_length, model_name)
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return None, None, 0.0, None, 0.0  # 新增返回训练时间

    if len(train_dataset) == 0 or len(dev_dataset) == 0:
        print("Skipping RTE due to empty dataset")
        return None, None, 0.0, None, 0.0  # 新增返回训练时间

    # 创建数据加载器
    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 初始化模型
    try:
        model = RTEModel(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None, None, 0.0, None, 0.0  # 新增返回训练时间

    # 优化器设置
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": lr
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr * 0.1
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    # 学习率调度器
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # 混合精度训练
    if use_amp:
        scaler = GradScaler()

    # 早停设置
    best_accuracy = 0.0
    patience = 3
    patience_counter = 0

    # 记录训练历史
    history = {
        'train_loss': [],
        'val_accuracy': []
    }

    # 训练循环
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            optimizer.zero_grad()

            # 准备输入数据
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            # 混合精度训练
            if use_amp:
                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = nn.CrossEntropyLoss()(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # 计算平均训练损失并记录
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].cpu().numpy()

                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        # 计算准确率并记录
        if len(all_labels) > 0:
            accuracy = accuracy_score(all_labels, all_preds)
        else:
            accuracy = 0.0
        history['val_accuracy'].append(accuracy)

        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # 早停检查
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            torch.save(model.state_dict(), "rte_best_model.pt")
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{patience} epochs")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 计算并打印总训练时间（新增代码）
    end_time = time.time()
    total_training_time = end_time - start_time
    minutes = int(total_training_time // 60)
    seconds = int(total_training_time % 60)
    print(f"\nTotal training time: {minutes} minutes {seconds} seconds")

    # 绘制训练曲线
    plot_path = plot_training_curves(history, model_name)

    # 加载最佳模型
    try:
        model.load_state_dict(torch.load("rte_best_model.pt"))
        print(f"Loaded best model with accuracy: {best_accuracy:.4f}")
    except Exception as e:
        print(f"Error loading best model: {e}")

    # 最终评估
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Final Evaluation"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].cpu().numpy()

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    if len(all_labels) > 0:
        final_accuracy = accuracy_score(all_labels, all_preds)
    else:
        final_accuracy = 0.0

    print(f"\nFinal Validation Accuracy: {final_accuracy:.4f}")

    # 返回总训练时间（新增）
    return model, tokenizer, final_accuracy, history, total_training_time


def generate_rte_submission(model, tokenizer, data_dir, max_length=192):
    """为RTE任务生成GLUE提交文件"""
    print("\nGenerating submission for RTE task...")

    if model is None:
        print("No model available for prediction")
        return None

    # 创建测试数据集
    try:
        test_dataset = RTEDataset(data_dir, 'test', max_length, tokenizer.name_or_path)
    except Exception as e:
        print(f"Error creating test dataset: {e}")
        return None

    if len(test_dataset) == 0:
        print("Skipping submission due to empty test dataset")
        return None

    # 创建数据加载器
    loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )

    # 生成预测
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
            all_preds.extend(preds)

    # 创建提交目录
    os.makedirs("glue_submissions", exist_ok=True)
    submission_file = os.path.join("glue_submissions", "RTE.tsv")

    # 保存预测结果
    try:
        with open(submission_file, 'w', encoding='utf-8') as f:
            for pred in all_preds:
                f.write(f"{pred}\n")

        print(f"Saved {len(all_preds)} predictions to {submission_file}")
        return submission_file
    except Exception as e:
        print(f"Error saving submission file: {e}")
        return None


if __name__ == "__main__":
    # 配置参数
    DATA_DIR = "./glue"  # GLUE数据集路径

    # 设置重试会话
    session = create_retry_session()
    requests.sessions.Session = session

    # 训练优化的RTE模型（接收训练时间返回值）
    rte_model, tokenizer, accuracy, history, total_time = train_rte_model(
        data_dir=DATA_DIR,
        model_name="roberta-base",
        epochs=10,
        batch_size=8,
        lr=1e-5,
        max_length=192
    )

    # 生成提交文件
    if rte_model:
        submission_file = generate_rte_submission(
            rte_model, tokenizer, DATA_DIR
        )

        print("\nRTE Training Complete!")
        print(f"Validation Accuracy: {accuracy:.4f}")
        # 打印总训练时间
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        print(f"Total training time: {minutes} minutes {seconds} seconds")
        if submission_file:
            print(f"Submission file: {submission_file}")
    else:
        print("RTE training failed")