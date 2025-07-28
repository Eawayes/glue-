import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.metrics import r2_score
from tqdm import tqdm
import re
import logging
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
import string
import warnings
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import random

# 忽略警告
warnings.filterwarnings('ignore')

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rte_training_optimized.log')
    ]
)
logger = logging.getLogger(__name__)

# 环境配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 配置GloVe路径
GLOVE_PATH = "./glove.6B.300d.txt"
EMBEDDING_DIM = 300
MAX_LEN = 150  # 调整序列长度，避免过长导致过拟合
BATCH_SIZE = 32
EPOCHS = 50  # 减少训练轮次
LEARNING_RATE = 5e-5  # 调整学习率
DROPOUT_RATE = 0.6  # 增加dropout减轻过拟合
NHEAD = 4  # 保持多头注意力
NUM_TRANSFORMER_LAYERS = 1  # 减少Transformer层数
CNN_FILTERS = 64
CNN_KERNEL_SIZES = [2, 3, 4]  # 调整卷积核大小，捕捉更短的模式


def init_spark_session():
    """初始化Spark会话"""
    logger.info("Initializing Spark session...")
    try:
        spark = SparkSession.builder \
            .appName("RTE_DataLoader") \
            .master("local[1]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("INFO")
        logger.info("Spark session initialized successfully")
        return spark
    except Exception as e:
        logger.error(f"Failed to initialize Spark session: {e}")
        logger.error(traceback.format_exc())
        return None


class GloveEmbeddings:
    """加载和管理GloVe词向量"""

    def __init__(self, glove_path):
        self.word2idx = {}
        self.idx2word = []
        self.embeddings = []
        self._load_glove(glove_path)
        self.embedding_dim = EMBEDDING_DIM

    def _load_glove(self, path):
        logger.info("Loading GloVe embeddings...")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                logger.info(f"Found {len(lines)} lines in GloVe file")

                for i, line in enumerate(tqdm(lines, desc="Loading GloVe")):
                    values = line.split()
                    if len(values) < EMBEDDING_DIM + 1:
                        continue

                    word = values[0]
                    vector = np.asarray(values[1:1 + EMBEDDING_DIM], dtype='float32')

                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)
                    self.embeddings.append(vector)

            # 添加特殊token
            for token in ['<pad>', '<unk>']:
                self.word2idx[token] = len(self.word2idx)
                self.idx2word.append(token)
                self.embeddings.append(np.random.normal(scale=0.6, size=(EMBEDDING_DIM,)))

            self.embeddings = np.array(self.embeddings)
            logger.info(f"Loaded {len(self.embeddings)} vectors")
        except Exception as e:
            logger.error(f"Error loading GloVe: {e}")
            logger.error(traceback.format_exc())

    def get_embedding_matrix(self):
        return torch.tensor(self.embeddings, dtype=torch.float32)


# 英文停用词列表
ENGLISH_STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
    'wouldn', "wouldn't"
])

# 同义词替换表（用于数据增强）
SYNONYMS = {
    'good': ['great', 'fine', 'excellent'],
    'bad': ['poor', 'terrible', 'awful'],
    'happy': ['glad', 'pleased', 'joyful'],
    'sad': ['unhappy', 'sorrowful', 'depressed'],
    'say': ['state', 'declare', 'mention'],
    'think': ['believe', 'consider', 'reckon'],
    'go': ['move', 'travel', 'proceed'],
    'come': ['arrive', 'approach', 'reach'],
    'make': ['create', 'produce', 'build'],
    'take': ['seize', 'grasp', 'fetch']
}


def augment_text(tokens, prob=0.1):
    """简单的数据增强：随机替换同义词"""
    augmented = []
    for token in tokens:
        if token in SYNONYMS and random.random() < prob:
            # 随机选择一个同义词
            augmented.append(random.choice(SYNONYMS[token]))
        else:
            augmented.append(token)
    return augmented


def clean_text(text):
    try:
        text = str(text).lower().strip()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        tokens = [word for word in tokens if word not in ENGLISH_STOPWORDS]
        return tokens
    except Exception as e:
        logger.error(f"Error cleaning text: {text[:50]}... - {e}")
        return []


def load_data_with_spark(spark, data_dir, split='train'):
    """使用Spark加载数据"""
    logger.info(f"Loading {split} set with Spark...")
    try:
        file_path = os.path.join(data_dir, "RTE", f"{split}.tsv")
        logger.info(f"Looking for file at: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()

        df = spark.read \
            .option("sep", "\t") \
            .option("header", "true") \
            .csv(file_path)

        logger.info(f"Loaded {df.count()} raw rows for {split}")

        # 列名规范化
        if 'sentence1' not in df.columns:
            if 'text_a' in df.columns:
                df = df.withColumnRenamed("text_a", "sentence1") \
                    .withColumnRenamed("text_b", "sentence2")
                logger.info("Renamed columns: text_a->sentence1, text_b->sentence2")
            elif 'premise' in df.columns:
                df = df.withColumnRenamed("premise", "sentence1") \
                    .withColumnRenamed("hypothesis", "sentence2")
                logger.info("Renamed columns: premise->sentence1, hypothesis->sentence2")
            else:
                logger.warning(f"Required columns missing in {split} set")

        pandas_df = df.toPandas()
        logger.info(f"Converted to Pandas DataFrame with {len(pandas_df)} rows")

        # 清洗和过滤
        pandas_df = pandas_df[
            pandas_df['sentence1'].notnull() &
            pandas_df['sentence2'].notnull() &
            (pandas_df['sentence1'] != "") &
            (pandas_df['sentence2'] != "")
            ]
        logger.info(f"After filtering: {len(pandas_df)} rows")

        # 标签处理
        if 'label' in pandas_df.columns:
            label_mapping = {"entailment": 1, "not_entailment": 0, "0": 0, "1": 1}
            pandas_df['label'] = pandas_df['label'].apply(
                lambda x: label_mapping.get(str(x).lower().strip(), 0)
            )
            logger.info(f"Label distribution:\n{pandas_df['label'].value_counts()}")

        # 文本清洗
        pandas_df['tokens1'] = pandas_df['sentence1'].apply(clean_text)
        pandas_df['tokens2'] = pandas_df['sentence2'].apply(clean_text)

        # 对训练集进行数据增强
        if split == 'train':
            logger.info("Applying data augmentation to training set...")
            # 随机选择20%的样本进行增强
            mask = np.random.choice([True, False], size=len(pandas_df), p=[0.2, 0.8])
            pandas_df.loc[mask, 'tokens1'] = pandas_df.loc[mask, 'tokens1'].apply(augment_text)
            pandas_df.loc[mask, 'tokens2'] = pandas_df.loc[mask, 'tokens2'].apply(augment_text)

        # 过滤空列表
        pandas_df = pandas_df[(pandas_df['tokens1'].str.len() > 0) & (pandas_df['tokens2'].str.len() > 0)]
        logger.info(f"Final {split} set size: {len(pandas_df)}")

        return pandas_df
    except Exception as e:
        logger.error(f"Error loading {split} data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


class RTEDataset(Dataset):
    """自定义RTE数据集"""

    def __init__(self, premise, hypothesis, labels, word2idx):
        self.premise = premise
        self.hypothesis = hypothesis
        self.labels = labels
        self.word2idx = word2idx
        self.unk_token = word2idx['<unk>']
        self.pad_token = word2idx['<pad>']
        logger.info(f"Created dataset with {len(self)} samples")

    def __len__(self):
        return len(self.premise)

    def __getitem__(self, idx):
        try:
            premise_seq = [self.word2idx.get(word, self.unk_token) for word in self.premise[idx]]
            hypothesis_seq = [self.word2idx.get(word, self.unk_token) for word in self.hypothesis[idx]]

            # 记录实际长度
            prem_len = min(len(premise_seq), MAX_LEN)
            hyp_len = min(len(hypothesis_seq), MAX_LEN)

            # 截断
            premise_seq = premise_seq[:MAX_LEN]
            hypothesis_seq = hypothesis_seq[:MAX_LEN]

            return {
                'premise': torch.tensor(premise_seq, dtype=torch.long),
                'prem_len': prem_len,
                'hypothesis': torch.tensor(hypothesis_seq, dtype=torch.long),
                'hyp_len': hyp_len,
                'label': torch.tensor(self.labels[idx], dtype=torch.long) if self.labels is not None else -1
            }
        except Exception as e:
            logger.error(f"Error getting item {idx}: {e}")
            return {
                'premise': torch.zeros(1, dtype=torch.long),
                'prem_len': 1,
                'hypothesis': torch.zeros(1, dtype=torch.long),
                'hyp_len': 1,
                'label': torch.tensor(-1, dtype=torch.long)
            }


def collate_fn(batch):
    """自定义批处理函数"""
    premise = [item['premise'] for item in batch]
    prem_len = torch.tensor([item['prem_len'] for item in batch])
    hypothesis = [item['hypothesis'] for item in batch]
    hyp_len = torch.tensor([item['hyp_len'] for item in batch])
    labels = [item['label'] for item in batch]

    # 动态填充
    premise_padded = pad_sequence(premise, batch_first=True, padding_value=0)
    hypothesis_padded = pad_sequence(hypothesis, batch_first=True, padding_value=0)

    return {
        'premise': premise_padded,
        'prem_len': prem_len,
        'hypothesis': hypothesis_padded,
        'hyp_len': hyp_len,
        'label': torch.stack(labels) if labels[0] is not None else None
    }


class PositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        pos_embeddings = self.position_embeddings(positions)
        return x + pos_embeddings


class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器"""

    def __init__(self, input_dim, num_filters, kernel_sizes, dropout=0.1):
        super(CNNFeatureExtractor, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=num_filters,
                          kernel_size=k, padding=(k - 1) // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for k in kernel_sizes
        ])
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.layer_norm = nn.LayerNorm(num_filters * len(kernel_sizes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch_size, features, seq_len]

        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)
            pooled = self.max_pool(conv_out).squeeze(-1)
            conv_outputs.append(pooled)

        cnn_features = torch.cat(conv_outputs, dim=1)
        cnn_features = self.layer_norm(cnn_features)
        cnn_features = self.dropout(cnn_features)
        return cnn_features


class EnhancedLSTMTransformerCNNModel(nn.Module):
    """优化后的混合模型"""

    def __init__(self, embedding_matrix, hidden_dim=256, num_layers=2, dropout=DROPOUT_RATE):
        super(EnhancedLSTMTransformerCNNModel, self).__init__()
        self.embedding_dim = embedding_matrix.size(1)
        self.hidden_dim = hidden_dim

        # 嵌入层
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.embedding_dropout = nn.Dropout(dropout / 2)

        # 双向LSTM编码器（减少层数）
        self.premise_lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.hypothesis_lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Transformer编码器层（减少层数）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=NHEAD,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_TRANSFORMER_LAYERS
        )

        # 位置编码
        self.positional_encoding = PositionalEncoding(hidden_dim * 2, max_len=MAX_LEN)

        # CNN特征提取器
        self.premise_cnn = CNNFeatureExtractor(
            input_dim=hidden_dim * 2,
            num_filters=CNN_FILTERS,
            kernel_sizes=CNN_KERNEL_SIZES,
            dropout=dropout / 2
        )

        self.hypothesis_cnn = CNNFeatureExtractor(
            input_dim=hidden_dim * 2,
            num_filters=CNN_FILTERS,
            kernel_sizes=CNN_KERNEL_SIZES,
            dropout=dropout / 2
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # 分类器（简化结构）
        self.classifier = nn.Sequential(
            nn.Linear((hidden_dim * 2 + CNN_FILTERS * len(CNN_KERNEL_SIZES)) * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 2)
        )
        logger.info(
            f"Optimized model initialized with hidden_dim={hidden_dim}, layers={num_layers}")
        logger.info(f"Transformer layers: {NUM_TRANSFORMER_LAYERS}, heads: {NHEAD}")

    def forward(self, premise, prem_len, hypothesis, hyp_len):
        # 嵌入层
        prem_emb = self.embedding(premise)
        prem_emb = self.embedding_dropout(prem_emb)

        hypo_emb = self.embedding(hypothesis)
        hypo_emb = self.embedding_dropout(hypo_emb)

        # LSTM编码
        packed_prem = pack_padded_sequence(prem_emb, prem_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_hypo = pack_padded_sequence(hypo_emb, hyp_len.cpu(), batch_first=True, enforce_sorted=False)

        prem_out, _ = self.premise_lstm(packed_prem)
        hypo_out, _ = self.hypothesis_lstm(packed_hypo)

        prem_out, _ = pad_packed_sequence(prem_out, batch_first=True)
        hypo_out, _ = pad_packed_sequence(hypo_out, batch_first=True)

        # 位置编码
        prem_out = self.positional_encoding(prem_out)
        hypo_out = self.positional_encoding(hypo_out)

        # Transformer编码
        prem_mask = self._generate_padding_mask(prem_len, premise.size(1))
        hypo_mask = self._generate_padding_mask(hyp_len, hypothesis.size(1))

        prem_trans = self.transformer_encoder(prem_out, src_key_padding_mask=prem_mask)
        hypo_trans = self.transformer_encoder(hypo_out, src_key_padding_mask=hypo_mask)

        # CNN特征提取
        prem_cnn_features = self.premise_cnn(prem_trans)
        hypo_cnn_features = self.hypothesis_cnn(hypo_trans)

        # 注意力机制
        prem_attn = self.attention(prem_trans).squeeze(-1)
        prem_attn = prem_attn.masked_fill(prem_mask, float('-inf'))
        prem_attn = torch.softmax(prem_attn, dim=1).unsqueeze(-1)
        prem_rep = torch.sum(prem_trans * prem_attn, dim=1)

        hypo_attn = self.attention(hypo_trans).squeeze(-1)
        hypo_attn = hypo_attn.masked_fill(hypo_mask, float('-inf'))
        hypo_attn = torch.softmax(hypo_attn, dim=1).unsqueeze(-1)
        hypo_rep = torch.sum(hypo_trans * hypo_attn, dim=1)

        # 结合CNN特征
        prem_rep = torch.cat([prem_rep, prem_cnn_features], dim=1)
        hypo_rep = torch.cat([hypo_rep, hypo_cnn_features], dim=1)

        # 特征交互
        diff = torch.abs(prem_rep - hypo_rep)
        mult = prem_rep * hypo_rep

        # 组合特征
        combined = torch.cat([prem_rep, hypo_rep, diff, mult], dim=1)

        # 分类
        logits = self.classifier(combined)
        return logits

    def _generate_padding_mask(self, lengths, max_len):
        mask = torch.zeros((lengths.size(0), max_len), dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            if length < max_len:
                mask[i, length:] = True
        return mask


def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """优化后的训练函数"""
    logger.info("Starting model training...")

    total_start_time = time.time()
    epoch_times = []

    # 计算类别权重
    labels = []
    for batch in train_loader:
        if batch['label'] is not None:
            labels.extend(batch['label'].cpu().numpy())

    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 优化器（添加权重衰减）
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=3, verbose=True, min_lr=1e-6)

    best_acc = 0
    best_f1 = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'epoch_times': []
    }

    # 早停机制参数
    patience = 8
    counter = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            try:
                optimizer.zero_grad()

                premise = batch['premise'].to(device)
                prem_len = batch['prem_len'].to(device)
                hypothesis = batch['hypothesis'].to(device)
                hyp_len = batch['hyp_len'].to(device)
                labels = batch['label'].to(device)

                logits = model(premise, prem_len, hypothesis, hyp_len)
                loss = criterion(logits, labels)

                loss.backward()
                # 梯度裁剪
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            except Exception as e:
                logger.error(f"Training error: {e}")
                continue

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # 验证阶段
        val_acc, val_loss, val_metrics = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_acc)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])

        # 记录epoch时间
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        history['epoch_times'].append(epoch_time)

        logger.info(f"Epoch {epoch + 1} - Time: {epoch_time:.2f}s, Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                    f"F1: {val_metrics['f1']:.4f}")

        # 保存最佳模型
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_acc = val_acc
            torch.save(model.state_dict(), "rte_best_hybrid_model_optimized.pt")
            logger.info(f"Saved new best model with F1: {best_f1:.4f}, Acc: {best_acc:.4f}")
            counter = 0  # 重置早停计数器
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    total_time = time.time() - total_start_time

    # 加载最佳模型
    try:
        model.load_state_dict(torch.load("rte_best_hybrid_model_optimized.pt"))
        final_acc, final_loss, final_metrics = evaluate_model(model, val_loader, criterion)
        logger.info(f"\nTraining complete! Total training time: {total_time:.2f} seconds")
        logger.info(f"Average epoch time: {np.mean(epoch_times):.2f} seconds")
        logger.info(f"Best validation F1: {best_f1:.4f}, Accuracy: {best_acc:.4f}")
        return model, best_acc, history, total_time
    except Exception as e:
        logger.error(f"Error loading best model: {e}")
        return model, best_acc, history, total_time


def evaluate_model(model, data_loader, criterion=None):
    """模型评估函数"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            try:
                premise = batch['premise'].to(device)
                prem_len = batch['prem_len'].to(device)
                hypothesis = batch['hypothesis'].to(device)
                hyp_len = batch['hyp_len'].to(device)
                labels = batch['label'].to(device)

                logits = model(premise, prem_len, hypothesis, hyp_len)
                preds = torch.argmax(logits, dim=1)

                if criterion:
                    loss = criterion(logits, labels)
                    total_loss += loss.item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                continue

    if len(all_labels) == 0:
        logger.error("No labels for evaluation!")
        return 0.0, 0.0, {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'r2': 0.0,
            'report': "",
            'confusion_matrix': None
        }

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    r2 = r2_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    avg_loss = total_loss / len(data_loader) if criterion and len(data_loader) > 0 else 0.0

    return acc, avg_loss, {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'r2': r2,
        'report': report,
        'confusion_matrix': conf_matrix
    }


def generate_submission(model, test_loader):
    """生成预测结果"""
    logger.info("Generating submission...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            try:
                premise = batch['premise'].to(device)
                prem_len = batch['prem_len'].to(device)
                hypothesis = batch['hypothesis'].to(device)
                hyp_len = batch['hyp_len'].to(device)

                logits = model(premise, prem_len, hypothesis, hyp_len)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                all_preds.extend([0] * batch['premise'].size(0))

    os.makedirs("glue_submissions", exist_ok=True)
    submission_path = os.path.join("glue_submissions", "RTE_optimized.tsv")

    try:
        with open(submission_path, 'w') as f:
            f.write("index\tprediction\n")
            for i, pred in enumerate(all_preds):
                f.write(f"{i}\t{pred}\n")

        logger.info(f"Saved submission file to {submission_path}")
        return submission_path
    except Exception as e:
        logger.error(f"Error saving submission: {e}")
        return None


def plot_training_history(history, model_name="Optimized_LSTM_Transformer_CNN_RTE"):
    """绘制训练历史图表"""
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    # F1分数曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['val_f1'], label='F1 Score', color='purple')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    # 精确率-召回率曲线
    plt.subplot(2, 2, 4)
    plt.plot(history['val_precision'], label='Precision', color='blue')
    plt.plot(history['val_recall'], label='Recall', color='red')
    plt.title('Precision and Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join("plots", f"{model_name}_training_history.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved training history plot to {plot_path}")

    return plot_path


def plot_confusion_matrix(conf_matrix, classes, model_name="Optimized_LSTM_Transformer_CNN_RTE"):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plot_path = os.path.join("plots", f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved confusion matrix to {plot_path}")

    return plot_path


def print_performance_metrics(metrics, total_time, epoch_times):
    """打印详细的性能指标"""
    print("\n" + "=" * 70)
    print("Performance Metrics Summary")
    print("=" * 70)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"R2 Score: {metrics['r2']:.4f}")
    print("\nTraining Time Statistics:")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average epoch time: {np.mean(epoch_times):.2f} seconds")
    print(f"Longest epoch: {np.max(epoch_times):.2f} seconds")
    print(f"Shortest epoch: {np.min(epoch_times):.2f} seconds")
    print("\nClassification Report:")
    print(metrics['report'])
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])


def main():
    DATA_DIR = "./glue"
    logger.info("Starting optimized RTE training pipeline with LSTM+Transformer+CNN")

    pipeline_start_time = time.time()

    # 初始化Spark会话
    spark = init_spark_session()
    if spark is None:
        logger.error("Failed to initialize Spark. Exiting.")
        return

    # 加载GloVe词向量
    logger.info("Loading GloVe embeddings...")
    glove = GloveEmbeddings(GLOVE_PATH)
    if len(glove.embeddings) == 0:
        logger.error("Failed to load GloVe embeddings. Exiting.")
        spark.stop()
        return

    embedding_matrix = glove.get_embedding_matrix().to(device)

    # 加载数据
    logger.info("Loading datasets...")
    try:
        train_df = load_data_with_spark(spark, DATA_DIR, 'train')
        val_df = load_data_with_spark(spark, DATA_DIR, 'dev')
        test_df = load_data_with_spark(spark, DATA_DIR, 'test')
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        spark.stop()
        return

    # 检查数据是否为空
    if len(train_df) == 0 or len(val_df) == 0:
        logger.error("No data loaded. Exiting.")
        spark.stop()
        return

    logger.info(f"Train samples: {len(train_df)}, Dev samples: {len(val_df)}, Test samples: {len(test_df)}")

    # 创建数据集
    logger.info("Creating datasets...")
    train_dataset = RTEDataset(
        train_df['tokens1'].tolist(),
        train_df['tokens2'].tolist(),
        train_df['label'].tolist(),
        glove.word2idx
    )

    val_dataset = RTEDataset(
        val_df['tokens1'].tolist(),
        val_df['tokens2'].tolist(),
        val_df['label'].tolist(),
        glove.word2idx
    )

    test_dataset = RTEDataset(
        test_df['tokens1'].tolist(),
        test_df['tokens2'].tolist(),
        None,
        glove.word2idx
    )

    # 创建数据加载器
    logger.info("Creating data loaders with dynamic padding...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 初始化优化后的模型
    logger.info("Initializing optimized LSTM+Transformer+CNN model...")
    model = EnhancedLSTMTransformerCNNModel(
        embedding_matrix,
        hidden_dim=256,  # 减小隐藏层维度
        num_layers=2,  # 减少LSTM层数
        dropout=DROPOUT_RATE
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    trained_model, acc, history, training_time = train_model(model, train_loader, val_loader)

    # 最终评估
    logger.info("Performing final evaluation...")
    final_acc, final_loss, final_metrics = evaluate_model(trained_model, val_loader)

    logger.info(f"\nFinal Validation Accuracy: {final_acc:.4f}")
    logger.info(f"Final F1 Score: {final_metrics['f1']:.4f}")
    logger.info(f"Final Precision: {final_metrics['precision']:.4f}")
    logger.info(f"Final Recall: {final_metrics['recall']:.4f}")
    logger.info(f"R2 Score: {final_metrics['r2']:.4f}")
    logger.info("\nClassification Report:\n" + final_metrics['report'])

    # 打印性能指标
    print_performance_metrics({
        'accuracy': final_acc,
        'precision': final_metrics['precision'],
        'recall': final_metrics['recall'],
        'f1': final_metrics['f1'],
        'r2': final_metrics['r2'],
        'report': final_metrics['report'],
        'confusion_matrix': final_metrics['confusion_matrix']
    }, training_time, history['epoch_times'])

    # 可视化
    logger.info("Generating visualizations...")
    plot_training_history(history)

    if final_metrics['confusion_matrix'] is not None:
        plot_confusion_matrix(
            final_metrics['confusion_matrix'],
            classes=["Not Entailment", "Entailment"]
        )

    # 生成提交文件
    if acc > 0.60:
        submission_path = generate_submission(trained_model, test_loader)
        if submission_path:
            logger.info(f"Submission saved to {submission_path}")
    else:
        logger.warning(f"Training accuracy {acc:.4f} too low, skipping submission")

    # 停止Spark会话
    spark.stop()

    # 计算总运行时间
    pipeline_time = time.time() - pipeline_start_time
    logger.info(f"Spark session stopped")
    logger.info(f"Total pipeline time: {pipeline_time:.2f} seconds")
    logger.info("Optimized training pipeline completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
