"""
inference/engine.py ── 云端 API 推理引擎
=========================================================
本地不再加载任何模型文件，所有推理通过 HTTP 请求
调用云端 FastAPI 服务（腾讯云 SCF / 自建服务器均可）。

对外接口与原版完全一致：
    model_manager.predict(key, payloads) -> list[dict]
    model_manager.available              -> list[str]
    model_manager.status()               -> dict
    model_manager.init_app(app)

因此 pages.py / api.py / 所有模板均无需任何修改。
=========================================================
"""
from __future__ import annotations

import requests
from typing import List, Dict, Any


# ── 各 label 的展示元数据 ─────────────────────────────────────
LABEL_META: Dict[str, Dict[str, Any]] = {
    "norm"          : {"cn": "正常流量", "icon": "✅", "risk": 0},
    "sqli"          : {"cn": "SQL注入",  "icon": "🔴", "risk": 3},
    "xss"           : {"cn": "XSS攻击",  "icon": "🟠", "risk": 2},
    "cmdi"          : {"cn": "命令注入", "icon": "🟡", "risk": 3},
    "path-traversal": {"cn": "路径穿越", "icon": "🟣", "risk": 2},
}


def _enrich(raw_result: dict, model_key: str) -> dict:
    """
    将云端 API 返回的单条 result 补充为 Flask 模板所需的完整字段。
    云端返回：payload / label / label_cn / confidence / all_probs
    本地强制覆盖：icon / risk（始终从本地 LABEL_META 取，忽略云端值）

    注意：云端不同版本的 api_server.py 可能使用文本标签（如 [HIGH]）
    作为 icon 字段，必须在此处强制使用本地 emoji，防止透传到模板。
    """
    label = raw_result.get("label", "norm")
    # 始终从本地元数据取，与云端 icon 字段完全解耦
    meta  = LABEL_META.get(label, {"cn": label, "icon": "❓", "risk": 0})
    return {
        "payload"   : raw_result.get("payload", ""),
        "label"     : label,
        "label_cn"  : raw_result.get("label_cn") or meta["cn"],
        "icon"      : meta["icon"],    # 强制本地值，永不使用云端 icon 字段
        "risk"      : meta["risk"],    # 同上
        "confidence": raw_result.get("confidence", 0.0),
        "all_probs" : raw_result.get("all_probs", {}),
        "model"     : model_key,
    }


class ModelManager:
    """
    通过 HTTP 调用云端推理 API，对上层完全透明地替换本地模型加载。

    配置项（config.py / 环境变量）：
        API_BASE_URL   云端服务根地址，如 http://localhost:9000
        API_TIMEOUT    单次请求超时秒数（默认 15）
    """

    _SUPPORTED = ["lgbm", "textcnn"]

    def __init__(self):
        self._base_url: str  = ""
        self._timeout:  int  = 15
        self._ok:       bool = False
        self._err:      str  = ""

    def init_app(self, app):
        self._base_url = app.config.get("API_BASE_URL", "").rstrip("/")
        self._timeout  = app.config.get("API_TIMEOUT", 15)

        if not self._base_url:
            self._err = "API_BASE_URL 未配置"
            app.logger.error(f"[ModelManager] {self._err}")
            return

        # 启动时健康检查
        try:
            resp = requests.get(
                f"{self._base_url}/health",
                timeout=self._timeout,
            )
            if resp.ok:
                self._ok = True
                app.logger.info(
                    f"[ModelManager] 云端 API 连接成功 -> {self._base_url}"
                )
            else:
                self._err = f"健康检查失败，HTTP {resp.status_code}"
                app.logger.warning(f"[ModelManager] {self._err}")
        except Exception as e:
            self._err = f"无法连接云端 API：{e}"
            app.logger.warning(f"[ModelManager] {self._err}")

    @property
    def available(self) -> List[str]:
        return self._SUPPORTED if self._ok else []

    def status(self) -> Dict[str, Any]:
        return {
            "api_url": self._base_url,
            "online" : self._ok,
            "loaded" : self.available,
            "failed" : {} if self._ok else {"api": self._err},
        }

    def predict(self, key: str, payloads) -> List[Dict[str, Any]]:
        """
        调用云端 /predict/batch，返回与本地版格式完全一致的结果列表。
        """
        if isinstance(payloads, str):
            payloads = [payloads]
        if not payloads:
            return []
        if key not in self._SUPPORTED:
            raise ValueError(f"不支持的模型：'{key}'，可选：{self._SUPPORTED}")
        if not self._base_url:
            raise RuntimeError("API_BASE_URL 未配置")

        url  = f"{self._base_url}/predict/batch"
        body = {"payloads": payloads, "model": key}

        try:
            resp = requests.post(url, json=body, timeout=self._timeout)
            resp.raise_for_status()
        except requests.Timeout:
            raise RuntimeError(
                f"云端 API 超时（>{self._timeout}s），请检查网络或增大 API_TIMEOUT"
            )
        except requests.ConnectionError:
            raise RuntimeError(
                f"无法连接云端 API：{self._base_url}"
            )
        except requests.HTTPError as e:
            raise RuntimeError(f"云端 API 返回错误：{e}")

        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"云端推理失败：{data}")

        return [_enrich(r, key) for r in data.get("results", [])]


# 全局单例
model_manager = ModelManager()
