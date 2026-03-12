"""
train.py ── 第一阶段：模型全流程实现
==================================================
数据集  : HttpParamsDataset
任务    : 多分类（norm / sqli / xss / cmdi / path-traversal）
模型①  : TF-IDF (字符级+词级+手工特征) + LightGBM
模型②  : TextCNN（字符级 Embedding）
输出    : outputs/ 目录下所有模型文件及评估图表

运行    : python train.py
==================================================
"""
# Python 3.9 兼容类型注解
from __future__ import annotations

import os, re, pickle, warnings
import numpy  as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")          # 无 GUI 环境必须设置
import matplotlib.pyplot as plt
import seaborn as sns

from urllib.parse import unquote
from sklearn.preprocessing        import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.utils.class_weight   import compute_class_weight
from sklearn.model_selection      import train_test_split

import lightgbm as lgb
import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════
# 0. 全局配置
# ════════════════════════════════════════════════
SEED        = 42
TRAIN_CSV   = "payload_train.csv"
TEST_CSV    = "payload_test.csv"
FULL_CSV    = "payload_full.csv"
TEXT_COL    = "payload"
LABEL_COL   = "attack_type"
OUT_DIR     = "outputs"
MAX_LEN     = 200          # TextCNN 序列截断长度
EPOCHS      = 30
BATCH_SIZE  = 256

np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Info] 设备: {DEVICE}")


# ════════════════════════════════════════════════
# 1. 数据加载
# ════════════════════════════════════════════════
def load_data():
    def _read(path):
        df = pd.read_csv(path, quotechar='"')
        df.columns = df.columns.str.strip().str.lower()
        return df

    if os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV):
        tr = _read(TRAIN_CSV)
        te = _read(TEST_CSV)
    else:
        print(f"[Data] 未找到 train/test，从 {FULL_CSV} 8:2 划分")
        df = _read(FULL_CSV)
        tr, te = train_test_split(
            df, test_size=0.2, random_state=SEED, stratify=df[LABEL_COL]
        )

    print(f"[Data] 训练 {len(tr)} 条 | 测试 {len(te)} 条")
    print(f"[Data] 训练集标签分布：\n{tr[LABEL_COL].value_counts()}\n")
    return tr, te


# ════════════════════════════════════════════════
# 2. 预处理（与推理引擎完全一致）
# ════════════════════════════════════════════════
def preprocess(text: str) -> str:
    """3 轮 URL 解码 → HTML 实体还原 → 小写 → 压缩空白"""
    if not isinstance(text, str):
        return ""
    for _ in range(3):
        d = unquote(text)
        if d == text:
            break
        text = d
    text = (text.replace("&lt;", "<").replace("&gt;", ">")
                .replace("&amp;", "&").replace("&quot;", '"')
                .replace("&#039;", "'"))
    return re.sub(r"\s+", " ", text.lower()).strip()


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TEXT_COL]  = df[TEXT_COL].fillna("").astype(str).apply(preprocess)
    df[LABEL_COL] = df[LABEL_COL].str.strip().str.lower()
    return df


# ════════════════════════════════════════════════
# 3. 手工数值特征（19 维）
# ════════════════════════════════════════════════
def numeric_feats(texts: list) -> sp.csr_matrix:
    rows = []
    for t in texts:
        rows.append([
            len(t),
            t.count("'"),        t.count('"'),     t.count("<"),
            t.count(">"),        t.count("/"),      t.count("\\"),
            t.count("("),        t.count("--"),     t.count("/*"),
            t.count("../"),      t.count("script"), t.count("select"),
            t.count("union"),    t.count("insert"), t.count("drop"),
            t.count("exec"),
            int(bool(re.search(r"\bor\b",  t))),
            int(bool(re.search(r"\band\b", t))),
        ])
    return sp.csr_matrix(np.array(rows, dtype=np.float32))


# ════════════════════════════════════════════════
# 4. TF-IDF 特征构建
# ════════════════════════════════════════════════
def build_tfidf(tr_texts: list, te_texts: list):
    print("[Feature] 构建 TF-IDF 特征...")

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        max_features=50_000, sublinear_tf=True, min_df=2,
    )
    word_tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=30_000, sublinear_tf=True, min_df=1,
        token_pattern=r"(?u)\S+",
    )

    Xtr = sp.hstack([
        char_tfidf.fit_transform(tr_texts),
        word_tfidf.fit_transform(tr_texts),
        numeric_feats(tr_texts),
    ])
    Xte = sp.hstack([
        char_tfidf.transform(te_texts),
        word_tfidf.transform(te_texts),
        numeric_feats(te_texts),
    ])
    print(f"[Feature] 训练维度: {Xtr.shape}")
    return Xtr, Xte, char_tfidf, word_tfidf


# ════════════════════════════════════════════════
# 5. 标签编码
# ════════════════════════════════════════════════
def encode_labels(tr_lbl: list, te_lbl: list):
    le = LabelEncoder()
    le.fit(tr_lbl)
    y_tr = le.transform(tr_lbl)
    y_te = le.transform(te_lbl)
    print(f"[Label] 类别: {list(le.classes_)}")
    return y_tr, y_te, le


# ════════════════════════════════════════════════
# 6. 通用评估 + 混淆矩阵可视化
# ════════════════════════════════════════════════
def evaluate(y_true, y_pred, le: LabelEncoder, name: str) -> dict:
    labels = le.classes_
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true,   y_pred, average="macro", zero_division=0)
    f1m  = f1_score(y_true,       y_pred, average="macro", zero_division=0)
    f1w  = f1_score(y_true,       y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*56}\n  {name} — 评估结果\n{'='*56}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision(macro): {prec:.4f}")
    print(f"  Recall(macro)   : {rec:.4f}")
    print(f"  F1(macro)       : {f1m:.4f}")
    print(f"  F1(weighted)    : {f1w:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=labels)}")

    # 混淆矩阵（原始 + 归一化各一张）
    cm     = confusion_matrix(y_true, y_pred)
    cm_n   = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in [
        (ax1, cm,   f"{name} 混淆矩阵（原始）",    "d"),
        (ax2, cm_n, f"{name} 混淆矩阵（归一化）", ".2f"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.set_title(title)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"cm_{name.lower().replace(' ','_')}.png")
    plt.savefig(fname, dpi=150); plt.close()
    print(f"[Eval] 混淆矩阵 → {fname}")
    return {"acc": acc, "prec": prec, "rec": rec, "f1_mac": f1m, "f1_wt": f1w}


# ════════════════════════════════════════════════
# 7. LightGBM 训练
# ════════════════════════════════════════════════
def train_lgbm(Xtr, y_tr, Xte, y_te, le: LabelEncoder):
    print(f"\n{'='*56}\n  LightGBM 训练\n{'='*56}")

    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    sw      = np.array([dict(zip(classes, weights))[y] for y in y_tr])

    params = {
        "objective"        : "multiclass",
        "num_class"        : len(le.classes_),
        "metric"           : "multi_logloss",
        "learning_rate"    : 0.05,
        "num_leaves"       : 127,
        "min_child_samples": 5,
        "feature_fraction" : 0.8,
        "bagging_fraction" : 0.8,
        "bagging_freq"     : 5,
        "lambda_l1"        : 0.1,
        "lambda_l2"        : 0.1,
        "verbose"          : -1,
        "n_jobs"           : -1,
        "seed"             : SEED,
    }
    ds_tr = lgb.Dataset(Xtr, label=y_tr, weight=sw)
    ds_te = lgb.Dataset(Xte, label=y_te, reference=ds_tr)

    model = lgb.train(
        params, ds_tr, num_boost_round=500,
        valid_sets=[ds_tr, ds_te],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(30, verbose=True),
            lgb.log_evaluation(50),
        ],
    )
    y_pred = np.argmax(model.predict(Xte), axis=1)
    return model, y_pred


def plot_importance(model, char_tf, word_tf, top_n=30):
    """绘制 LightGBM Top-N 特征重要性"""
    imp      = model.feature_importance(importance_type="gain")
    c_names  = [f"c:{k}" for k, _ in sorted(char_tf.vocabulary_.items(), key=lambda x: x[1])]
    w_names  = [f"w:{k}" for k, _ in sorted(word_tf.vocabulary_.items(), key=lambda x: x[1])]
    n_names  = ["len","'",'"',"<",">","/","\\","(","--","/*",
                 "../","script","select","union","insert","drop","exec","or","and"]
    df = pd.DataFrame({"feature": c_names + w_names + n_names, "imp": imp})
    df = df.nlargest(top_n, "imp")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df["feature"][::-1], df["imp"][::-1], color="steelblue")
    ax.set_title(f"LightGBM Top-{top_n} 特征重要性（Gain）")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lgbm_feature_importance.png"), dpi=150)
    plt.close()
    print("[Info] 特征重要性图已保存")


# ════════════════════════════════════════════════
# 8. TextCNN 训练
# ════════════════════════════════════════════════
class CharVocab:
    PAD, UNK = 0, 1
    def __init__(self):
        chars = [chr(i) for i in range(32, 127)]
        self.c2i = {c: i + 2 for i, c in enumerate(chars)}
        self.c2i["<PAD>"] = self.PAD
        self.c2i["<UNK>"] = self.UNK
        self.size = len(self.c2i)

    def encode(self, text: str, max_len: int) -> list:
        ids = [self.c2i.get(c, self.UNK) for c in text[:max_len]]
        return ids + [self.PAD] * (max_len - len(ids))


class PayloadDS(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.X = [torch.tensor(vocab.encode(t, max_len), dtype=torch.long)
                  for t in texts]
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 kernels=(2, 3, 4, 5), filters=128, dropout=0.5):
        super().__init__()
        self.emb    = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs  = nn.ModuleList([nn.Conv1d(embed_dim, filters, k) for k in kernels])
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(filters * len(kernels), num_classes)

    def forward(self, x):
        e = self.emb(x).permute(0, 2, 1)
        p = [F.adaptive_max_pool1d(F.relu(c(e)), 1).squeeze(2) for c in self.convs]
        return self.fc(self.drop(torch.cat(p, dim=1)))


def train_textcnn(tr_texts, y_tr, te_texts, y_te, le: LabelEncoder):
    print(f"\n{'='*56}\n  TextCNN 训练\n{'='*56}")

    vocab = CharVocab()
    nc    = len(le.classes_)
    dl_tr = DataLoader(PayloadDS(tr_texts, y_tr, vocab, MAX_LEN),
                       batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    dl_te = DataLoader(PayloadDS(te_texts,  y_te, vocab, MAX_LEN),
                       batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TextCNN(vocab.size, 64, nc).to(DEVICE)

    cls    = np.unique(y_tr)
    cw     = compute_class_weight("balanced", classes=cls, y=y_tr)
    crit   = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float).to(DEVICE)
    )
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_f1, best_pred = 0.0, None
    hist_loss, hist_f1 = [], []

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            total_loss += loss.item() * len(xb)
        sched.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in dl_te:
                preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
        vf1 = f1_score(y_te, preds, average="macro", zero_division=0)

        avg = total_loss / len(dl_tr.dataset)
        hist_loss.append(avg); hist_f1.append(vf1)

        if vf1 > best_f1:
            best_f1   = vf1
            best_pred = np.array(preds)
            torch.save(model.state_dict(),
                       os.path.join(OUT_DIR, "textcnn_best.pt"))

        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:>3}/{EPOCHS} | Loss {avg:.4f} | Val F1 {vf1:.4f}")

    # 训练曲线
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(hist_loss, color="royalblue", label="Train Loss")
    ax2.plot(hist_f1,   color="tomato",    label="Val Macro-F1")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="royalblue")
    ax2.set_ylabel("Macro F1", color="tomato")
    ax1.set_title("TextCNN 训练曲线")
    fig.legend(loc="upper right", bbox_to_anchor=(.88, .88))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "textcnn_training_curve.png"), dpi=150)
    plt.close()
    print(f"[TextCNN] 最佳 Val Macro-F1: {best_f1:.4f}")
    return best_pred


# ════════════════════════════════════════════════
# 9. 模型对比图
# ════════════════════════════════════════════════
def compare(results: dict):
    df = pd.DataFrame(results).T
    print(f"\n{'='*56}\n  模型性能汇总\n{'='*56}")
    print(df.to_string())
    df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"))

    fig, ax = plt.subplots(figsize=(10, 4))
    x, w = np.arange(len(df)), 0.15
    cols  = ["#4e9af1", "#f97316", "#22c55e", "#a855f7", "#e11d48"]
    for i, col in enumerate(df.columns):
        ax.bar(x + i * w, df[col], w, label=col, color=cols[i % len(cols)])
    ax.set_xticks(x + w * 1.5); ax.set_xticklabels(df.index)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
    ax.set_title("模型性能对比"); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()


# ════════════════════════════════════════════════
# 10. 主流程
# ════════════════════════════════════════════════
def main():
    # ── 加载 & 预处理 ─────────────────────────
    tr_df, te_df = load_data()
    tr_df = preprocess_df(tr_df)
    te_df = preprocess_df(te_df)

    tr_texts = tr_df[TEXT_COL].tolist()
    te_texts = te_df[TEXT_COL].tolist()
    tr_lbls  = tr_df[LABEL_COL].tolist()
    te_lbls  = te_df[LABEL_COL].tolist()

    # 过滤测试集中训练集未见标签（防止 LabelEncoder 报错）
    seen = set(tr_lbls)
    mask = [l in seen for l in te_lbls]
    te_texts = [t for t, m in zip(te_texts, mask) if m]
    te_lbls  = [l for l, m in zip(te_lbls,  mask) if m]

    # ── 标签编码 ──────────────────────────────
    y_tr, y_te, le = encode_labels(tr_lbls, te_lbls)

    # ══════════════════════════════════════════
    # LightGBM
    # ══════════════════════════════════════════
    Xtr, Xte, char_tf, word_tf = build_tfidf(tr_texts, te_texts)
    lgbm_model, y_pred_lgb     = train_lgbm(Xtr, y_tr, Xte, y_te, le)
    res_lgbm = evaluate(y_te, y_pred_lgb, le, "LightGBM")

    lgbm_model.save_model(os.path.join(OUT_DIR, "lgbm_model.txt"))
    print("[Save] lgbm_model.txt")
    plot_importance(lgbm_model, char_tf, word_tf)

    # ══════════════════════════════════════════
    # TextCNN
    # ══════════════════════════════════════════
    y_pred_cnn = train_textcnn(tr_texts, y_tr, te_texts, y_te, le)
    res_cnn    = evaluate(y_te, y_pred_cnn, le, "TextCNN")

    # ══════════════════════════════════════════
    # 保存推理引擎所需的辅助文件（必须）
    # ══════════════════════════════════════════
    for fname, obj in [
        ("label_encoder.pkl", le),
        ("char_tfidf.pkl",    char_tf),
        ("word_tfidf.pkl",    word_tf),
    ]:
        with open(os.path.join(OUT_DIR, fname), "wb") as f:
            pickle.dump(obj, f)
        print(f"[Save] {fname}")

    # ══════════════════════════════════════════
    # 对比汇总
    # ══════════════════════════════════════════
    compare({"LightGBM": res_lgbm, "TextCNN": res_cnn})

    print("\n─── outputs/ 文件清单 ───")
    for fn in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, fn))
        print(f"  {fn:<42} {sz/1024:>8.1f} KB")
    print("\n✅  第一阶段全部完成！")


if __name__ == "__main__":
    main()
