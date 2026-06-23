# !/user/bin/env python3
# -*- coding: utf-8 -*-
"""
api_server.py — WAD 模型 API 服务
====================================================
基于 FastAPI 封装 LightGBM 和 TextCNN 推理接口

启动:
    pip install fastapi uvicorn
    python api_server.py

访问:
    API 文档  → http://localhost:8000/docs
    接口测试  → http://localhost:8000/redoc

接口列表:
    GET  /health                    健康检查
    POST /predict                   单条检测
    POST /predict/batch             批量检测（JSON 列表）
====================================================
"""
from __future__ import annotations

import time
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── 导入推理函数 ────────────────────────────────────────────
from predict import predict_lgbm, predict_textcnn

# ════════════════════════════════════════════════════════════
# 应用初始化
# ════════════════════════════════════════════════════════════
app = FastAPI(
    title="WAD · Web 漏洞攻击检测 API",
    description="""
基于 HttpParamsDataset 训练的多分类检测模型，支持：

- **LightGBM**：TF-IDF 特征 + 梯度提升，高精度，推荐首选
- **TextCNN**：字符级 Embedding + CNN，深度学习方案

**攻击类型**：norm（正常）/ sqli（SQL注入）/ xss（XSS）/ cmdi（命令注入）/ path-traversal（路径穿越）
    """,
    version="1.0.0",
)

# 允许跨域（方便前端或其他服务调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════
# 请求 / 响应数据模型
# ════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    payload: str = Field(..., description="HTTP 请求参数值", example="' OR 1=1 --")
    model: str   = Field("lgbm", description="模型选择：lgbm 或 textcnn", example="lgbm")


class BatchRequest(BaseModel):
    payloads: List[str] = Field(
        ...,
        description="payload 列表，最多 500 条",
        example=["' OR 1=1 --", "<script>alert(1)</script>", "normal text"],
    )
    model: str = Field("lgbm", description="模型选择：lgbm 或 textcnn")


class ProbsDetail(BaseModel):
    norm:           float
    sqli:           float
    xss:            float
    cmdi:           float
    path_traversal: float = Field(alias="path-traversal")

    class Config:
        populate_by_name = True


class PredictResult(BaseModel):
    payload:    str
    label:      str   = Field(description="预测标签")
    label_cn:   str   = Field(description="中文标签")
    confidence: float = Field(description="最高类别置信度（0~1）")
    all_probs:  dict  = Field(description="全部 5 个类别的概率分布")
    model:      str   = Field(description="使用的模型")


class PredictResponse(BaseModel):
    success:     bool
    result:      PredictResult
    elapsed_ms:  float = Field(description="推理耗时（毫秒）")


class BatchResponse(BaseModel):
    success:      bool
    total:        int
    attack_count: int
    normal_count: int
    elapsed_ms:   float
    results:      List[PredictResult]


# ════════════════════════════════════════════════════════════
# 接口实现
# ════════════════════════════════════════════════════════════

@app.get("/health", summary="健康检查")
def health():
    """确认服务正常运行"""
    return {"status": "ok", "service": "WAD API", "version": "1.0.0"}


@app.post("/predict", response_model=PredictResponse, summary="单条 Payload 检测")
def predict(req: PredictRequest):
    """
    对单条 HTTP 请求参数进行攻击类型检测。

    - **payload**：需要检测的参数值
    - **model**：`lgbm`（默认）或 `textcnn`
    """
    if not req.payload.strip():
        raise HTTPException(status_code=400, detail="payload 不能为空")
    if req.model not in ("lgbm", "textcnn"):
        raise HTTPException(status_code=400, detail="model 须为 lgbm 或 textcnn")

    t0 = time.perf_counter()
    try:
        fn   = predict_lgbm if req.model == "lgbm" else predict_textcnn
        pred = fn([req.payload])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败：{e}")

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    pred["model"] = req.model

    return PredictResponse(
        success=True,
        result=PredictResult(**pred),
        elapsed_ms=elapsed,
    )


@app.post("/predict/batch", response_model=BatchResponse, summary="批量 Payload 检测")
def predict_batch(req: BatchRequest):
    """
    批量检测，最多 500 条。

    - **payloads**：payload 字符串列表
    - **model**：`lgbm`（默认）或 `textcnn`
    """
    if not req.payloads:
        raise HTTPException(status_code=400, detail="payloads 不能为空")
    if len(req.payloads) > 500:
        raise HTTPException(status_code=400, detail="单次最多 500 条")
    if req.model not in ("lgbm", "textcnn"):
        raise HTTPException(status_code=400, detail="model 须为 lgbm 或 textcnn")

    t0 = time.perf_counter()
    try:
        fn    = predict_lgbm if req.model == "lgbm" else predict_textcnn
        preds = fn(req.payloads)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败：{e}")

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    for p in preds:
        p["model"] = req.model

    attack = sum(1 for p in preds if p["label"] != "norm")
    return BatchResponse(
        success=True,
        total=len(preds),
        attack_count=attack,
        normal_count=len(preds) - attack,
        elapsed_ms=elapsed,
        results=[PredictResult(**p) for p in preds],
    )


# ════════════════════════════════════════════════════════════
# 启动
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,       # 生产环境关闭热重载
    )