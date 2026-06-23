"""
train.py ── Web漏洞攻击多分类检测系统
==================================================
数据集  : HttpParamsDataset（已预划分，禁止重新划分）
任务    : 5分类（norm / sqli / xss / cmdi / path-traversal）
模型①  : TF-IDF (字符级+词级+手工特征) + LightGBM
模型②  : TextCNN（字符级 Embedding）
输出    : outputs/ 目录下所有模型文件、评估图表、训练日志

运行    : python train.py
==================================================
"""
# Python 3.9 兼容类型注解
from __future__ import annotations

import os
import re
import json
import pickle
import warnings
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境必须设置
import matplotlib.pyplot as plt
import seaborn as sns

from urllib.parse import unquote
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════
# 0. 全局配置与可复现性设置
# ════════════════════════════════════════════════

# 固定随机种子，确保实验可复现
SEED = 42

# 设置Python随机种子
random.seed(SEED)

# 设置NumPy随机种子
np.random.seed(SEED)

# 设置PyTorch随机种子
torch.manual_seed(SEED)

# 设置CUDA随机种子（如果可用）
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# 禁用CUDA非确定性算法，确保结果可复现
torch.backends.cudnn.deterministic = True

# 禁用CUDA基准模式，避免因优化导致的不可复现性
torch.backends.cudnn.benchmark = False

# 数据集文件路径（已预划分，禁止重新划分）
TRAIN_CSV = "HttpParamsDataset/train_split.csv"
VAL_CSV = "HttpParamsDataset/val_split.csv"
TEST_CSV = "HttpParamsDataset/payload_test.csv"

# 特征列名
TEXT_COL = "payload"
LABEL_COL = "attack_type"

# 输出目录
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# TextCNN超参数
MAX_LEN = 200  # 序列截断长度
EPOCHS = 30  # 最大训练轮数
BATCH_SIZE = 256  # 批次大小
PATIENCE = 7  # 早停耐心值

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Info] 设备: {DEVICE}")
print(f"[Info] 随机种子: {SEED}")
print(f"[Info] CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[Info] CUDA版本: {torch.version.cuda}")
    print(f"[Info] GPU设备: {torch.cuda.get_device_name(0)}")


# ════════════════════════════════════════════════
# 1. 数据加载（严格使用预划分的3个文件）
# ════════════════════════════════════════════════
def load_data():
    """
    加载已预划分的数据集文件

    返回:
        train_df: 训练集DataFrame
        val_df: 验证集DataFrame
        test_df: 测试集DataFrame
    """
    print("[Data] 加载已预划分的数据集...")

    # 检查文件是否存在
    for filepath in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"数据集文件不存在: {filepath}")

    # 加载训练集
    train_df = pd.read_csv(TRAIN_CSV, quotechar='"')
    train_df.columns = train_df.columns.str.strip().str.lower()
    print(f"[Data] 训练集加载完成: {len(train_df)} 条")

    # 加载验证集
    val_df = pd.read_csv(VAL_CSV, quotechar='"')
    val_df.columns = val_df.columns.str.strip().str.lower()
    print(f"[Data] 验证集加载完成: {len(val_df)} 条")

    # 加载测试集
    test_df = pd.read_csv(TEST_CSV, quotechar='"')
    test_df.columns = test_df.columns.str.strip().str.lower()
    print(f"[Data] 测试集加载完成: {len(test_df)} 条")

    # 打印各类别分布
    print("\n[Data] 训练集类别分布:")
    print(train_df[LABEL_COL].value_counts().to_string())
    print("\n[Data] 验证集类别分布:")
    print(val_df[LABEL_COL].value_counts().to_string())
    print("\n[Data] 测试集类别分布:")
    print(test_df[LABEL_COL].value_counts().to_string())

    return train_df, val_df, test_df


# ════════════════════════════════════════════════
# 2. 预处理（与推理引擎完全一致）
# ════════════════════════════════════════════════
def preprocess(text: str) -> str:
    """
    文本预处理函数：3轮URL解码 → HTML实体还原 → 小写 → 压缩空白

    参数:
        text: 原始文本

    返回:
        处理后的文本
    """
    if not isinstance(text, str):
        return ""

    # 3轮URL解码（处理多层编码的情况）
    for _ in range(3):
        decoded = unquote(text)
        if decoded == text:
            break
        text = decoded

    # HTML实体还原
    text = (text.replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&amp;", "&")
                .replace("&quot;", '"')
                .replace("&#039;", "'"))

    # 小写转换并压缩空白字符
    return re.sub(r"\s+", " ", text.lower()).strip()


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    对DataFrame进行预处理

    参数:
        df: 原始DataFrame

    返回:
        处理后的DataFrame
    """
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).apply(preprocess)
    df[LABEL_COL] = df[LABEL_COL].str.strip().str.lower()
    return df


# ════════════════════════════════════════════════
# 3. 手工数值特征（19维）
# ════════════════════════════════════════════════
def numeric_feats(texts: list) -> sp.csr_matrix:
    """
    提取手工数值特征（19维）

    特征包括：
    - 文本长度
    - 特殊字符计数（引号、尖括号、斜杠等）
    - 攻击关键词计数（script、select、union等）
    - 布尔特征（or、and等）

    参数:
        texts: 文本列表

    返回:
        稀疏特征矩阵
    """
    rows = []
    for t in texts:
        features = [
            len(t),  # 文本长度
            t.count("'"),  # 单引号
            t.count('"'),  # 双引号
            t.count("<"),  # 左尖括号
            t.count(">"),  # 右尖括号
            t.count("/"),  # 斜杠
            t.count("\\"),  # 反斜杠
            t.count("("),  # 左括号
            t.count("--"),  # SQL注释
            t.count("/*"),  # SQL块注释
            t.count("../"),  # 路径遍历
            t.count("script"),  # XSS关键词
            t.count("select"),  # SQL关键词
            t.count("union"),  # SQL关键词
            t.count("insert"),  # SQL关键词
            t.count("drop"),  # SQL关键词
            t.count("exec"),  # 命令执行关键词
            int(bool(re.search(r"\bor\b", t))),  # SQL逻辑运算符
            int(bool(re.search(r"\band\b", t))),  # SQL逻辑运算符
        ]
        rows.append(features)
    return sp.csr_matrix(np.array(rows, dtype=np.float32))


# ════════════════════════════════════════════════
# 4. TF-IDF 特征构建
# ════════════════════════════════════════════════
def build_tfidf(train_texts: list, val_texts: list, test_texts: list):
    """
    构建TF-IDF特征（字符级 + 词级 + 手工特征）

    参数:
        train_texts: 训练集文本列表
        val_texts: 验证集文本列表
        test_texts: 测试集文本列表

    返回:
        X_train: 训练集特征矩阵
        X_val: 验证集特征矩阵
        X_test: 测试集特征矩阵
        char_tfidf: 字符级TF-IDF向量化器
        word_tfidf: 词级TF-IDF向量化器
    """
    print("[Feature] 构建 TF-IDF 特征...")

    # 字符级TF-IDF（捕获字符n-gram模式）
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),  # 2-5字符的n-gram
        max_features=50_000,  # 最大特征数
        sublinear_tf=True,  # 使用亚线性TF缩放
        min_df=2,  # 最小文档频率
    )

    # 词级TF-IDF（捕获词n-gram模式）
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),  # 1-2词的n-gram
        max_features=30_000,  # 最大特征数
        sublinear_tf=True,  # 使用亚线性TF缩放
        min_df=1,  # 最小文档频率
        token_pattern=r"(?u)\S+",  # 以非空白字符分割
    )

    # 在训练集上拟合并转换
    X_train = sp.hstack([
        char_tfidf.fit_transform(train_texts),
        word_tfidf.fit_transform(train_texts),
        numeric_feats(train_texts),
    ])

    # 在验证集和测试集上转换（不重新拟合）
    X_val = sp.hstack([
        char_tfidf.transform(val_texts),
        word_tfidf.transform(val_texts),
        numeric_feats(val_texts),
    ])

    X_test = sp.hstack([
        char_tfidf.transform(test_texts),
        word_tfidf.transform(test_texts),
        numeric_feats(test_texts),
    ])

    print(f"[Feature] 训练维度: {X_train.shape}")
    print(f"[Feature] 验证维度: {X_val.shape}")
    print(f"[Feature] 测试维度: {X_test.shape}")

    return X_train, X_val, X_test, char_tfidf, word_tfidf


# ════════════════════════════════════════════════
# 5. 标签编码
# ════════════════════════════════════════════════
def encode_labels(train_labels: list, val_labels: list, test_labels: list):
    """
    标签编码

    参数:
        train_labels: 训练集标签列表
        val_labels: 验证集标签列表
        test_labels: 测试集标签列表

    返回:
        y_train: 训练集编码后标签
        y_val: 验证集编码后标签
        y_test: 测试集编码后标签
        le: 标签编码器
    """
    le = LabelEncoder()

    # 在训练集上拟合
    le.fit(train_labels)

    # 转换所有标签
    y_train = le.transform(train_labels)
    y_val = le.transform(val_labels)
    y_test = le.transform(test_labels)

    print(f"[Label] 类别: {list(le.classes_)}")
    print(f"[Label] 类别数量: {len(le.classes_)}")

    return y_train, y_val, y_test, le


# ════════════════════════════════════════════════
# 6. 计算类别权重（处理数据不平衡）
# ════════════════════════════════════════════════
def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    计算类别权重，用于处理数据不平衡问题

    使用sklearn的compute_class_weight函数，基于"balanced"策略计算权重。
    权重公式：weight = n_samples / (n_classes * np.bincount(y))

    参数:
        y_train: 训练集标签数组

    返回:
        class_weight_dict: 类别权重字典 {类别索引: 权重}
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    print("\n[Weight] 类别权重:")
    for cls, weight in class_weight_dict.items():
        print(f"  类别 {cls}: {weight:.4f}")

    return class_weight_dict


# ════════════════════════════════════════════════
# 7. 通用评估 + 混淆矩阵可视化
# ════════════════════════════════════════════════
def evaluate(y_true, y_pred, le: LabelEncoder, name: str, save_dir: str = OUT_DIR) -> dict:
    """
    模型评估函数

    计算并打印各项评估指标，生成混淆矩阵图

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        le: 标签编码器
        name: 评估名称（如"LightGBM"、"TextCNN"）
        save_dir: 保存目录

    返回:
        metrics: 评估指标字典
    """
    labels = le.classes_

    # 计算各项指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # 打印评估结果
    print(f"\n{'='*60}")
    print(f"  {name} — 评估结果")
    print(f"{'='*60}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Precision(macro)  : {prec:.4f}")
    print(f"  Recall(macro)     : {rec:.4f}")
    print(f"  F1(macro)         : {f1_mac:.4f}")
    print(f"  F1(micro)         : {f1_micro:.4f}")
    print(f"  F1(weighted)      : {f1_wt:.4f}")

    # 打印详细分类报告
    print(f"\n{classification_report(y_true, y_pred, target_names=labels)}")

    # 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建混淆矩阵图（原始 + 归一化）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 原始混淆矩阵
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_xlabel("预测", fontsize=12)
    ax1.set_ylabel("真实", fontsize=12)
    ax1.set_title(f"{name} 混淆矩阵（原始）", fontsize=14)

    # 归一化混淆矩阵
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_xlabel("预测", fontsize=12)
    ax2.set_ylabel("真实", fontsize=12)
    ax2.set_title(f"{name} 混淆矩阵（归一化）", fontsize=14)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"cm_{name.lower().replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] 混淆矩阵 → {fname}")

    # 返回评估指标
    metrics = {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1_mac": f1_mac,
        "f1_micro": f1_micro,
        "f1_wt": f1_wt,
    }

    return metrics


# ════════════════════════════════════════════════
# 8. LightGBM 训练
# ════════════════════════════════════════════════
def train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test, le: LabelEncoder):
    """
    LightGBM模型训练

    使用TF-IDF特征训练LightGBM多分类模型，支持类别权重和早停机制

    参数:
        X_train: 训练集特征矩阵
        y_train: 训练集标签
        X_val: 验证集特征矩阵
        y_val: 验证集标签
        X_test: 测试集特征矩阵
        y_test: 测试集标签
        le: 标签编码器

    返回:
        model: 训练好的LightGBM模型
        y_pred_test: 测试集预测结果
        y_pred_val: 验证集预测结果
    """
    print(f"\n{'='*60}")
    print(f"  LightGBM 训练")
    print(f"{'='*60}")

    # 计算类别权重
    class_weight_dict = compute_class_weights(y_train)

    # 为每个样本分配权重
    sample_weights = np.array([class_weight_dict[y] for y in y_train])

    # LightGBM参数配置
    params = {
        "objective": "multiclass",
        "num_class": len(le.classes_),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "seed": SEED,
    }

    # 创建数据集
    ds_train = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)

    # 训练模型
    model = lgb.train(
        params,
        ds_train,
        num_boost_round=500,
        valid_sets=[ds_train, ds_val],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(30, verbose=True),  # 早停机制
            lgb.log_evaluation(50),  # 每50轮打印一次
        ],
    )

    # 验证集预测
    y_pred_val = np.argmax(model.predict(X_val), axis=1)

    # 测试集预测
    y_pred_test = np.argmax(model.predict(X_test), axis=1)

    # 保存模型
    model_path = os.path.join(OUT_DIR, "lgbm_model.txt")
    model.save_model(model_path)
    print(f"[Save] LightGBM模型 → {model_path}")

    return model, y_pred_test, y_pred_val


def plot_importance(model, char_tf, word_tf, top_n=30):
    """
    绘制LightGBM特征重要性图

    参数:
        model: LightGBM模型
        char_tf: 字符级TF-IDF向量化器
        word_tf: 词级TF-IDF向量化器
        top_n: 显示前N个重要特征
    """
    # 获取特征重要性
    importance = model.feature_importance(importance_type="gain")

    # 获取特征名称
    char_names = [f"c:{k}" for k, _ in sorted(char_tf.vocabulary_.items(), key=lambda x: x[1])]
    word_names = [f"w:{k}" for k, _ in sorted(word_tf.vocabulary_.items(), key=lambda x: x[1])]
    numeric_names = ["len", "'", '"', "<", ">", "/", "\\", "(", "--", "/*",
                     "../", "script", "select", "union", "insert", "drop", "exec", "or", "and"]

    # 创建特征重要性DataFrame
    df = pd.DataFrame({
        "feature": char_names + word_names + numeric_names,
        "importance": importance
    })
    df = df.nlargest(top_n, "importance")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制特征重要性图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="steelblue")
    ax.set_title(f"LightGBM Top-{top_n} 特征重要性（Gain）", fontsize=14)
    ax.set_xlabel("Importance", fontsize=12)
    plt.tight_layout()

    # 保存图片
    fname = os.path.join(OUT_DIR, "lgbm_feature_importance.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Info] 特征重要性图 → {fname}")


# ════════════════════════════════════════════════
# 9. TextCNN 训练
# ════════════════════════════════════════════════
class CharVocab:
    """字符级词汇表"""
    PAD, UNK = 0, 1

    def __init__(self):
        # ASCII可打印字符（32-126）
        chars = [chr(i) for i in range(32, 127)]
        self.c2i = {c: i + 2 for i, c in enumerate(chars)}
        self.c2i["<PAD>"] = self.PAD
        self.c2i["<UNK>"] = self.UNK
        self.size = len(self.c2i)

    def encode(self, text: str, max_len: int) -> list:
        """将文本编码为整数序列"""
        ids = [self.c2i.get(c, self.UNK) for c in text[:max_len]]
        return ids + [self.PAD] * (max_len - len(ids))


class PayloadDS(Dataset):
    """Payload数据集类"""

    def __init__(self, texts, labels, vocab, max_len):
        self.X = [torch.tensor(vocab.encode(t, max_len), dtype=torch.long)
                  for t in texts]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class TextCNN(nn.Module):
    """TextCNN模型"""

    def __init__(self, vocab_size, embed_dim, num_classes,
                 kernels=(2, 3, 4, 5), filters=128, dropout=0.5):
        super().__init__()
        # 嵌入层
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 多尺度卷积层
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, filters, k) for k in kernels])
        # Dropout层
        self.drop = nn.Dropout(dropout)
        # 全连接层
        self.fc = nn.Linear(filters * len(kernels), num_classes)

    def forward(self, x):
        # 嵌入：(batch, seq_len) -> (batch, embed_dim, seq_len)
        e = self.emb(x).permute(0, 2, 1)
        # 多尺度卷积 + 最大池化
        p = [F.adaptive_max_pool1d(F.relu(conv(e)), 1).squeeze(2) for conv in self.convs]
        # 拼接 + Dropout + 全连接
        return self.fc(self.drop(torch.cat(p, dim=1)))


def train_textcnn(train_texts, y_train, val_texts, y_val, test_texts, y_test, le: LabelEncoder):
    """
    TextCNN模型训练

    使用字符级嵌入的TextCNN模型进行训练，支持类别权重和早停机制

    参数:
        train_texts: 训练集文本列表
        y_train: 训练集标签
        val_texts: 验证集文本列表
        y_val: 验证集标签
        test_texts: 测试集文本列表
        y_test: 测试集标签
        le: 标签编码器

    返回:
        y_pred_test: 测试集预测结果
        y_pred_val: 验证集预测结果
        training_log: 训练日志
    """
    print(f"\n{'='*60}")
    print(f"  TextCNN 训练")
    print(f"{'='*60}")

    # 创建字符词汇表
    vocab = CharVocab()
    num_classes = len(le.classes_)

    # 创建数据加载器
    # 设置worker_init_fn以确保数据加载的可复现性
    def worker_init_fn(worker_id):
        np.random.seed(SEED + worker_id)
        random.seed(SEED + worker_id)

    train_dataset = PayloadDS(train_texts, y_train, vocab, MAX_LEN)
    val_dataset = PayloadDS(val_texts, y_val, vocab, MAX_LEN)
    test_dataset = PayloadDS(test_texts, y_test, vocab, MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    # 创建TextCNN模型
    model = TextCNN(vocab.size, 64, num_classes).to(DEVICE)

    # 计算类别权重
    class_weight_dict = compute_class_weights(y_train)
    class_weights = torch.tensor([class_weight_dict[i] for i in range(num_classes)],
                                 dtype=torch.float).to(DEVICE)

    # 损失函数（带类别权重）
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 学习率调度器（ReduceLROnPlateau）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控指标越大越好
        factor=0.5,  # 学习率衰减因子
        patience=3,  # 耐心值
        verbose=True,
    )

    # 训练日志
    training_log = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1_mac": [],
        "lr": [],
    }

    # 早停机制
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        # ========== 训练阶段 ==========
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item() * len(xb)
            _, predicted = outputs.max(1)
            total += yb.size(0)
            correct += predicted.eq(yb).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total

        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                outputs = model(xb)
                loss = criterion(outputs, yb)

                val_loss += loss.item() * len(xb)
                _, predicted = outputs.max(1)
                val_total += yb.size(0)
                val_correct += predicted.eq(yb).sum().item()
                val_preds.extend(predicted.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        # 计算验证集宏平均F1
        val_f1_mac = f1_score(y_val, val_preds, average="macro", zero_division=0)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 记录训练日志
        training_log["epoch"].append(epoch)
        training_log["train_loss"].append(train_loss)
        training_log["val_loss"].append(val_loss)
        training_log["train_acc"].append(train_acc)
        training_log["val_acc"].append(val_acc)
        training_log["val_f1_mac"].append(val_f1_mac)
        training_log["lr"].append(current_lr)

        # 打印训练信息
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F1: {val_f1_mac:.4f} | "
                  f"LR: {current_lr:.6f}")

        # 学习率调度
        scheduler.step(val_f1_mac)

        # 早停机制检查（基于验证集宏平均F1）
        if val_f1_mac > best_f1:
            best_f1 = val_f1_mac
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[Early Stopping] 在第 {epoch} 轮触发早停")
                print(f"[Early Stopping] 最佳验证F1: {best_f1:.4f} (第 {best_epoch} 轮)")
                break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 保存最佳模型
    model_path = os.path.join(OUT_DIR, "textcnn_best.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[Save] TextCNN最佳模型 → {model_path}")

    # ========== 测试集预测 ==========
    model.eval()
    test_preds = []

    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            outputs = model(xb)
            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())

    # 验证集预测（使用最佳模型）
    model.eval()
    val_preds_final = []

    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(DEVICE)
            outputs = model(xb)
            _, predicted = outputs.max(1)
            val_preds_final.extend(predicted.cpu().numpy())

    return np.array(test_preds), np.array(val_preds_final), training_log


def plot_training_curves(training_log: dict):
    """
    绘制训练曲线（损失曲线和F1分数曲线）

    参数:
        training_log: 训练日志字典
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    epochs = training_log["epoch"]

    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # 损失曲线
    ax1.plot(epochs, training_log["train_loss"], color="royalblue", label="Train Loss", linewidth=2)
    ax1.plot(epochs, training_log["val_loss"], color="tomato", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("训练损失曲线", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, training_log["train_acc"], color="royalblue", label="Train Acc", linewidth=2)
    ax2.plot(epochs, training_log["val_acc"], color="tomato", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("准确率曲线", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # F1分数曲线
    ax3.plot(epochs, training_log["val_f1_mac"], color="forestgreen", label="Val Macro-F1", linewidth=2)
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("F1 Score", fontsize=12)
    ax3.set_title("验证集宏平均F1曲线", fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    fname = os.path.join(OUT_DIR, "textcnn_training_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Info] 训练曲线 → {fname}")


# ════════════════════════════════════════════════
# 10. 模型对比图
# ════════════════════════════════════════════════
def compare_models(results: dict):
    """
    模型性能对比

    参数:
        results: 模型评估结果字典 {模型名: 评估指标}
    """
    # 创建DataFrame
    df = pd.DataFrame(results).T

    # 打印对比结果
    print(f"\n{'='*60}")
    print(f"  模型性能汇总")
    print(f"{'='*60}")
    print(df.to_string())

    # 保存到CSV
    csv_path = os.path.join(OUT_DIR, "model_comparison.csv")
    df.to_csv(csv_path)
    print(f"[Save] 模型对比 → {csv_path}")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制对比图
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    width = 0.15

    colors = ["#4e9af1", "#f97316", "#22c55e", "#a855f7", "#e11d48"]

    for i, col in enumerate(df.columns):
        ax.bar(x + i * width, df[col], width, label=col, color=colors[i % len(colors)])

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df.index, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("模型性能对比", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图片
    fname = os.path.join(OUT_DIR, "model_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Info] 模型对比图 → {fname}")


# ════════════════════════════════════════════════
# 11. 保存训练日志
# ════════════════════════════════════════════════
def save_training_logs(training_log: dict, lgbm_metrics: dict, textcnn_metrics: dict):
    """
    保存训练日志到JSON文件

    参数:
        training_log: TextCNN训练日志
        lgbm_metrics: LightGBM评估指标
        textcnn_metrics: TextCNN评估指标
    """
    logs = {
        "textcnn_training": training_log,
        "lgbm_metrics": lgbm_metrics,
        "textcnn_metrics": textcnn_metrics,
    }

    # 保存为JSON
    json_path = os.path.join(OUT_DIR, "training_logs.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
    print(f"[Save] 训练日志 → {json_path}")

    # 保存为CSV（便于后续画图）
    csv_path = os.path.join(OUT_DIR, "textcnn_training_log.csv")
    pd.DataFrame(training_log).to_csv(csv_path, index=False)
    print(f"[Save] TextCNN训练日志CSV → {csv_path}")


# ════════════════════════════════════════════════
# 12. 少数类性能分析
# ════════════════════════════════════════════════
def analyze_minority_classes(y_true, y_pred, le: LabelEncoder, model_name: str):
    """
    分析少数类（cmdi、path-traversal、xss）的性能表现

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        le: 标签编码器
        model_name: 模型名称
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} — 少数类性能分析")
    print(f"{'='*60}")

    labels = le.classes_
    minority_classes = ["cmdi", "path-traversal", "xss"]

    for cls_name in minority_classes:
        if cls_name not in labels:
            continue

        cls_idx = le.transform([cls_name])[0]

        # 计算该类的各项指标
        cls_true = (y_true == cls_idx).astype(int)
        cls_pred = (y_pred == cls_idx).astype(int)

        tp = np.sum((cls_true == 1) & (cls_pred == 1))
        fp = np.sum((cls_true == 0) & (cls_pred == 1))
        fn = np.sum((cls_true == 1) & (cls_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        support = np.sum(cls_true == 1)

        print(f"\n  类别: {cls_name}")
        print(f"    支持数: {support}")
        print(f"    精确率: {precision:.4f}")
        print(f"    召回率: {recall:.4f}")
        print(f"    F1分数: {f1:.4f}")

        # 改进建议
        if recall < 0.7:
            print(f"    ⚠️  召回率偏低，建议：")
            print(f"       - 增加类别权重")
            print(f"       - 数据增强（谨慎使用）")
            print(f"       - 调整决策阈值")
        if precision < 0.7:
            print(f"    ⚠️  精确率偏低，建议：")
            print(f"       - 增加负样本挖掘")
            print(f"       - 优化特征工程")


# ════════════════════════════════════════════════
# 13. 主流程
# ════════════════════════════════════════════════
def main():
    """
    主函数：完整的训练-验证-测试流水线
    """
    print("\n" + "="*60)
    print("  Web漏洞攻击多分类检测系统")
    print("="*60)

    # ========== 1. 加载数据 ==========
    print("\n[Step 1] 加载已预划分的数据集...")
    train_df, val_df, test_df = load_data()

    # ========== 2. 预处理 ==========
    print("\n[Step 2] 数据预处理...")
    train_df = preprocess_df(train_df)
    val_df = preprocess_df(val_df)
    test_df = preprocess_df(test_df)

    # 提取文本和标签
    train_texts = train_df[TEXT_COL].tolist()
    val_texts = val_df[TEXT_COL].tolist()
    test_texts = test_df[TEXT_COL].tolist()

    train_labels = train_df[LABEL_COL].tolist()
    val_labels = val_df[LABEL_COL].tolist()
    test_labels = test_df[LABEL_COL].tolist()

    # ========== 3. 标签编码 ==========
    print("\n[Step 3] 标签编码...")
    y_train, y_val, y_test, le = encode_labels(train_labels, val_labels, test_labels)

    # ========== 4. 特征工程 ==========
    print("\n[Step 4] 特征工程...")
    X_train, X_val, X_test, char_tfidf, word_tfidf = build_tfidf(
        train_texts, val_texts, test_texts
    )

    # ========== 5. LightGBM训练 ==========
    print("\n[Step 5] LightGBM模型训练...")
    lgbm_model, y_pred_lgb_test, y_pred_lgb_val = train_lgbm(
        X_train, y_train, X_val, y_val, X_test, y_test, le
    )

    # LightGBM评估
    print("\n[Step 5.1] LightGBM验证集评估:")
    res_lgbm_val = evaluate(y_val, y_pred_lgb_val, le, "LightGBM验证集")

    print("\n[Step 5.2] LightGBM测试集评估:")
    res_lgbm_test = evaluate(y_test, y_pred_lgb_test, le, "LightGBM测试集")

    # 保存LightGBM特征提取器
    for fname, obj in [
        ("char_tfidf.pkl", char_tfidf),
        ("word_tfidf.pkl", word_tfidf),
        ("label_encoder.pkl", le),
    ]:
        with open(os.path.join(OUT_DIR, fname), "wb") as f:
            pickle.dump(obj, f)
        print(f"[Save] {fname}")

    # 绘制特征重要性
    plot_importance(lgbm_model, char_tfidf, word_tfidf)

    # ========== 6. TextCNN训练 ==========
    print("\n[Step 6] TextCNN模型训练...")
    y_pred_cnn_test, y_pred_cnn_val, training_log = train_textcnn(
        train_texts, y_train,
        val_texts, y_val,
        test_texts, y_test,
        le
    )

    # TextCNN评估
    print("\n[Step 6.1] TextCNN验证集评估:")
    res_cnn_val = evaluate(y_val, y_pred_cnn_val, le, "TextCNN验证集")

    print("\n[Step 6.2] TextCNN测试集评估:")
    res_cnn_test = evaluate(y_test, y_pred_cnn_test, le, "TextCNN测试集")

    # 绘制训练曲线
    plot_training_curves(training_log)

    # ========== 7. 少数类性能分析 ==========
    print("\n[Step 7] 少数类性能分析...")
    analyze_minority_classes(y_test, y_pred_lgb_test, le, "LightGBM")
    analyze_minority_classes(y_test, y_pred_cnn_test, le, "TextCNN")

    # ========== 8. 模型对比 ==========
    print("\n[Step 8] 模型性能对比...")
    compare_models({
        "LightGBM验证集": res_lgbm_val,
        "LightGBM测试集": res_lgbm_test,
        "TextCNN验证集": res_cnn_val,
        "TextCNN测试集": res_cnn_test,
    })

    # ========== 9. 保存训练日志 ==========
    print("\n[Step 9] 保存训练日志...")
    save_training_logs(training_log, res_lgbm_test, res_cnn_test)

    # ========== 10. 打印输出文件清单 ==========
    print("\n" + "="*60)
    print("  outputs/ 文件清单")
    print("="*60)
    for fn in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, fn))
        print(f"  {fn:<45} {sz/1024:>8.1f} KB")

    print("\n✅ 全部完成！")
    print("="*60)


if __name__ == "__main__":
    main()
