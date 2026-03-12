# !/user/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py — WAD 推理接口（黑盒封装）
====================================================
对外只暴露两个函数：
    predict_lgbm(payloads)    使用 LightGBM 模型
    predict_textcnn(payloads) 使用 TextCNN 模型

调用示例：
    from predict import predict_lgbm, predict_textcnn

    results = predict_lgbm(["' OR 1=1 --", "hello world"])
    print(results[0])
    # {
    #   "payload"   : "' OR 1=1 --",
    #   "label"     : "sqli",
    #   "label_cn"  : "SQL注入",
    #   "confidence": 0.9952,
    #   "all_probs" : {
    #       "norm": 0.001, "sqli": 0.995,
    #       "xss": 0.002, "cmdi": 0.001, "path-traversal": 0.001
    #   }
    # }
====================================================
"""
from __future__ import annotations   # Python 3.9 兼容

import os
import re
import pickle
from typing import List, Dict, Any

import numpy  as np
import scipy.sparse as sp
from urllib.parse import unquote

# ── 模型文件目录（默认与本脚本同级的 outputs/）──────────────
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# ── 标签中文映射 ─────────────────────────────────────────────
_LABEL_CN = {
    "norm"          : "正常流量",
    "sqli"          : "SQL注入",
    "xss"           : "XSS攻击",
    "cmdi"          : "命令注入",
    "path-traversal": "路径穿越",
}

# ════════════════════════════════════════════════════════════
# 内部工具函数（调用方无需关心）
# ════════════════════════════════════════════════════════════

def _preprocess(text: str) -> str:
    """3 轮 URL 解码 → HTML 实体还原 → 小写 → 压缩空白"""
    if not isinstance(text, str):
        return ""
    for _ in range(3):
        decoded = unquote(text)
        if decoded == text:
            break
        text = decoded
    text = (text.replace("&lt;",   "<")
                .replace("&gt;",   ">")
                .replace("&amp;",  "&")
                .replace("&quot;", '"')
                .replace("&#039;", "'"))
    return re.sub(r"\s+", " ", text.lower()).strip()


def _numeric_feats(texts: List[str]) -> sp.csr_matrix:
    """19 维手工字符统计特征"""
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


def _build_result(raw: str, label: str, prob_row: np.ndarray,
                  pred_idx: int, classes: np.ndarray) -> Dict[str, Any]:
    """构造统一格式的返回字典"""
    return {
        "payload"   : raw,
        "label"     : label,
        "label_cn"  : _LABEL_CN.get(label, label),
        "confidence": round(float(prob_row[pred_idx]), 6),
        "all_probs" : {
            cls: round(float(prob_row[j]), 6)
            for j, cls in enumerate(classes)
        },
    }


# ════════════════════════════════════════════════════════════
# 懒加载：模型只在第一次调用时加载，之后复用
# ════════════════════════════════════════════════════════════

_lgbm_cache    = {}   # {"model", "le", "char_tf", "word_tf"}
_textcnn_cache = {}   # {"net", "le", "device"}


def _load_lgbm():
    if _lgbm_cache:
        return _lgbm_cache
    import lightgbm as lgb
    _lgbm_cache["model"]   = lgb.Booster(model_file=os.path.join(_MODEL_DIR, "lgbm_model.txt"))
    with open(os.path.join(_MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        _lgbm_cache["le"]      = pickle.load(f)
    with open(os.path.join(_MODEL_DIR, "char_tfidf.pkl"), "rb") as f:
        _lgbm_cache["char_tf"] = pickle.load(f)
    with open(os.path.join(_MODEL_DIR, "word_tfidf.pkl"), "rb") as f:
        _lgbm_cache["word_tf"] = pickle.load(f)
    return _lgbm_cache


def _load_textcnn():
    if _textcnn_cache:
        return _textcnn_cache

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    with open(os.path.join(_MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    # ── 字符词表（与训练完全一致）──────────────────────────
    chars = [chr(i) for i in range(32, 127)]
    char2id = {c: i + 2 for i, c in enumerate(chars)}
    char2id["<PAD>"] = 0
    char2id["<UNK>"] = 1
    vocab_size = len(char2id)

    # ── 网络结构（与训练完全一致）──────────────────────────
    class TextCNN(nn.Module):
        def __init__(self, vsz, edim, nc, ks=(2,3,4,5), nf=128, dr=0.5):
            super().__init__()
            self.emb   = nn.Embedding(vsz, edim, padding_idx=0)
            self.convs = nn.ModuleList([nn.Conv1d(edim, nf, k) for k in ks])
            self.drop  = nn.Dropout(dr)
            self.fc    = nn.Linear(nf * len(ks), nc)
        def forward(self, x):
            e = self.emb(x).permute(0, 2, 1)
            p = [F.adaptive_max_pool1d(F.relu(c(e)), 1).squeeze(2) for c in self.convs]
            return self.fc(self.drop(torch.cat(p, dim=1)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net    = TextCNN(vocab_size, 64, len(le.classes_)).to(device)
    state  = torch.load(
        os.path.join(_MODEL_DIR, "textcnn_best.pt"),
        map_location=device,
        weights_only=True,   # 消除 PyTorch 2.x FutureWarning
    )
    net.load_state_dict(state)
    net.eval()

    _textcnn_cache["net"]     = net
    _textcnn_cache["le"]      = le
    _textcnn_cache["char2id"] = char2id
    _textcnn_cache["device"]  = device
    _textcnn_cache["torch"]   = torch
    return _textcnn_cache


# ════════════════════════════════════════════════════════════
# 对外接口
# ════════════════════════════════════════════════════════════

def predict_lgbm(payloads: List[str]) -> List[Dict[str, Any]]:
    """
    使用 LightGBM 模型进行攻击类型检测。

    参数
    ----
    payloads : list[str]
        HTTP 请求参数值列表，可以是单条或多条。

    返回
    ----
    list[dict]，每条结果包含：
        payload    : str   原始输入
        label      : str   预测标签（norm/sqli/xss/cmdi/path-traversal）
        label_cn   : str   中文标签（正常流量/SQL注入/...）
        confidence : float 最高类别的置信度（0~1）
        all_probs  : dict  全部类别的概率分布
    """
    if isinstance(payloads, str):
        payloads = [payloads]

    cache     = _load_lgbm()
    model     = cache["model"]
    le        = cache["le"]
    char_tf   = cache["char_tf"]
    word_tf   = cache["word_tf"]

    processed = [_preprocess(t) for t in payloads]
    X = sp.hstack([
        char_tf.transform(processed),
        word_tf.transform(processed),
        _numeric_feats(processed),
    ])
    probs    = model.predict(X)                    # shape (N, C)
    pred_ids = np.argmax(probs, axis=1)
    labels   = le.inverse_transform(pred_ids)

    return [
        _build_result(payloads[i], labels[i], probs[i], pred_ids[i], le.classes_)
        for i in range(len(payloads))
    ]


def predict_textcnn(payloads: List[str]) -> List[Dict[str, Any]]:
    """
    使用 TextCNN 模型进行攻击类型检测。

    参数 / 返回格式与 predict_lgbm 完全相同。
    """
    if isinstance(payloads, str):
        payloads = [payloads]

    cache   = _load_textcnn()
    net     = cache["net"]
    le      = cache["le"]
    char2id = cache["char2id"]
    device  = cache["device"]
    torch   = cache["torch"]

    max_len = 200

    def encode(text):
        ids = [char2id.get(c, 1) for c in text[:max_len]]
        return ids + [0] * (max_len - len(ids))

    processed = [_preprocess(t) for t in payloads]
    x = torch.tensor(
        [encode(t) for t in processed],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1).cpu().numpy()

    pred_ids = np.argmax(probs, axis=1)
    labels   = le.inverse_transform(pred_ids)

    return [
        _build_result(payloads[i], labels[i], probs[i], pred_ids[i], le.classes_)
        for i in range(len(payloads))
    ]


# ════════════════════════════════════════════════════════════
# 命令行快速测试
# python predict.py
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    test_cases = [
        "campello, el",
        "' OR 1=1 --",
        "<script>alert(1)</script>",
        "; cat /etc/passwd",
        "../../../../etc/shadow",
    ]

    print("=" * 56)
    print("  LightGBM 预测")
    print("=" * 56)
    for r in predict_lgbm(test_cases):
        print(f"  [{r['label']:>14}] {r['confidence']:.4f}  {r['payload'][:40]}")

    print()
    print("=" * 56)
    print("  TextCNN 预测")
    print("=" * 56)
    for r in predict_textcnn(test_cases):
        print(f"  [{r['label']:>14}] {r['confidence']:.4f}  {r['payload'][:40]}")