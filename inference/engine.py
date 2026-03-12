"""
inference/engine.py ── 模型推理引擎
修复②：包名 inference，无任何 PyPI 同名包冲突
修复③：from __future__ import annotations，兼容 Python 3.9 类型注解
修复④：torch.load weights_only=True，消除 PyTorch 2.x FutureWarning
"""
from __future__ import annotations

import os, re, pickle
from typing import List, Dict, Any

import numpy as np
import scipy.sparse as sp
from urllib.parse import unquote


# ════════════════════════════════════════════════
# 工具函数（与 train.py 逐字对齐）
# ════════════════════════════════════════════════
def _preprocess(text: str) -> str:
    if not isinstance(text, str): return ""
    for _ in range(3):
        d = unquote(text)
        if d == text: break
        text = d
    text = (text.replace("&lt;", "<").replace("&gt;", ">")
                .replace("&amp;", "&").replace("&quot;", '"')
                .replace("&#039;", "'"))
    return re.sub(r"\s+", " ", text.lower()).strip()


def _num_feats(texts: List[str]) -> sp.csr_matrix:
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


LABEL_META: Dict[str, Dict[str, Any]] = {
    "norm"          : {"cn": "正常流量", "icon": "✅", "risk": 0},
    "sqli"          : {"cn": "SQL注入",  "icon": "🔴", "risk": 3},
    "xss"           : {"cn": "XSS攻击",  "icon": "🟠", "risk": 2},
    "cmdi"          : {"cn": "命令注入", "icon": "🟡", "risk": 3},
    "path-traversal": {"cn": "路径穿越", "icon": "🟣", "risk": 2},
}


def _fmt(raw, label, prob_row, idx, classes, model_name) -> Dict[str, Any]:
    meta = LABEL_META.get(label, {"cn": label, "icon": "❓", "risk": 0})
    return {
        "payload"   : raw,
        "label"     : label,
        "label_cn"  : meta["cn"],
        "icon"      : meta["icon"],
        "risk"      : meta["risk"],
        "confidence": round(float(prob_row[idx]), 6),
        "all_probs" : {c: round(float(prob_row[j]), 6) for j, c in enumerate(classes)},
        "model"     : model_name,
    }


# ════════════════════════════════════════════════
# LightGBM 推理器
# ════════════════════════════════════════════════
class LGBMPredictor:
    MODEL_KEY = "lgbm"

    def __init__(self, model_dir: str):
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=os.path.join(model_dir, "lgbm_model.txt"))
        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            self.le = pickle.load(f)
        with open(os.path.join(model_dir, "char_tfidf.pkl"), "rb") as f:
            self.char_tf = pickle.load(f)
        with open(os.path.join(model_dir, "word_tfidf.pkl"), "rb") as f:
            self.word_tf = pickle.load(f)

    def predict(self, payloads: List[str]) -> List[Dict[str, Any]]:
        proc = [_preprocess(t) for t in payloads]
        X = sp.hstack([self.char_tf.transform(proc),
                       self.word_tf.transform(proc),
                       _num_feats(proc)])
        probs = self.model.predict(X)
        idxs  = np.argmax(probs, axis=1)
        lbls  = self.le.inverse_transform(idxs)
        return [_fmt(payloads[i], lbls[i], probs[i], idxs[i],
                     self.le.classes_, self.MODEL_KEY)
                for i in range(len(payloads))]


# ════════════════════════════════════════════════
# TextCNN 推理器
# ════════════════════════════════════════════════
class _Vocab:
    PAD, UNK = 0, 1
    def __init__(self):
        chars = [chr(i) for i in range(32, 127)]
        self.c2i = {c: i + 2 for i, c in enumerate(chars)}
        self.c2i.update({"<PAD>": 0, "<UNK>": 1})
        self.size = len(self.c2i)
    def encode(self, t, n=200):
        ids = [self.c2i.get(c, self.UNK) for c in t[:n]]
        return ids + [self.PAD] * (n - len(ids))


class TextCNNPredictor:
    MODEL_KEY = "textcnn"

    def __init__(self, model_dir: str):
        import torch, torch.nn as nn, torch.nn.functional as F
        self._torch  = torch
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocab   = _Vocab()
        self.max_len = 200
        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            self.le = pickle.load(f)
        nc = len(self.le.classes_)

        class _Net(nn.Module):
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

        self.net = _Net(self.vocab.size, 64, nc).to(self.device)
        # weights_only=True 消除 PyTorch 2.x FutureWarning
        state = torch.load(os.path.join(model_dir, "textcnn_best.pt"),
                           map_location=self.device, weights_only=True)
        self.net.load_state_dict(state)
        self.net.eval()

    def predict(self, payloads: List[str]) -> List[Dict[str, Any]]:
        torch = self._torch
        proc  = [_preprocess(t) for t in payloads]
        x     = torch.tensor([self.vocab.encode(t, self.max_len) for t in proc],
                              dtype=torch.long).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.net(x), dim=1).cpu().numpy()
        idxs = np.argmax(probs, axis=1)
        lbls = self.le.inverse_transform(idxs)
        return [_fmt(payloads[i], lbls[i], probs[i], idxs[i],
                     self.le.classes_, self.MODEL_KEY)
                for i in range(len(payloads))]


# ════════════════════════════════════════════════
# 统一管理器（全局单例）
# ════════════════════════════════════════════════
class ModelManager:
    def __init__(self):
        self._preds: Dict[str, Any] = {}
        self._errs:  Dict[str, str] = {}

    def init_app(self, app):
        d = app.config["MODEL_DIR"]
        for cls in (LGBMPredictor, TextCNNPredictor):
            k = cls.MODEL_KEY
            try:
                self._preds[k] = cls(d)
                app.logger.info(f"[MM] {k} 加载成功")
            except Exception as e:
                self._errs[k] = str(e)
                app.logger.warning(f"[MM] {k} 加载失败: {e}")

    @property
    def available(self) -> List[str]:
        return list(self._preds.keys())

    def predict(self, key: str, payloads) -> List[Dict[str, Any]]:
        if isinstance(payloads, str): payloads = [payloads]
        if key not in self._preds:
            raise ValueError(f"模型 '{key}' 未加载：{self._errs.get(key, '未知')}")
        return self._preds[key].predict(payloads)

    def status(self) -> Dict[str, Any]:
        return {"loaded": list(self._preds.keys()), "failed": self._errs}


model_manager = ModelManager()
