"""blueprints/api.py ── REST API"""
from __future__ import annotations

import json, uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import func
from extensions import db
from models.record import DetectionRecord
from inference.engine import model_manager

api_bp = Blueprint("api", __name__, url_prefix="/api")

def _ok(data=None, **kw):  return jsonify({"success": True,  "data": data, **kw})
def _err(msg, code=400):   return jsonify({"success": False, "error": msg}), code


@api_bp.get("/status")
def status():
    return _ok({"models": model_manager.status(),
                "total":  DetectionRecord.query.count(),
                "time":   datetime.utcnow().isoformat()+"Z"})


@api_bp.post("/detect")
def detect_one():
    """body: {"payload":"...", "model":"lgbm"}"""
    b   = request.get_json(silent=True) or {}
    pay = str(b.get("payload","")).strip()
    mod = b.get("model","lgbm")
    if not pay: return _err("payload 不能为空")
    try:
        pred = model_manager.predict(mod, [pay])[0]
    except ValueError as e: return _err(str(e))
    except Exception  as e:
        current_app.logger.exception(e); return _err("内部错误", 500)
    rec = DetectionRecord(payload=pay, model_used=mod, label=pred["label"],
                          confidence=pred["confidence"],
                          all_probs=json.dumps(pred["all_probs"]), source="api")
    db.session.add(rec); db.session.commit()
    return _ok({**pred, "record_id": rec.id})


@api_bp.post("/detect/batch")
def detect_batch():
    """body: {"payloads":[...], "model":"lgbm"}"""
    b    = request.get_json(silent=True) or {}
    pays = b.get("payloads", [])
    mod  = b.get("model", "lgbm")
    maxn = current_app.config["BATCH_MAX"]
    if not isinstance(pays, list) or not pays: return _err("payloads 须为非空列表")
    if len(pays) > maxn: return _err(f"单次最多 {maxn} 条")
    try:
        preds = model_manager.predict(mod, pays)
    except ValueError as e: return _err(str(e))
    except Exception  as e:
        current_app.logger.exception(e); return _err("内部错误", 500)
    bid = str(uuid.uuid4())
    db.session.bulk_save_objects([
        DetectionRecord(payload=p["payload"], model_used=mod, label=p["label"],
                        confidence=p["confidence"],
                        all_probs=json.dumps(p["all_probs"]),
                        source="api", batch_id=bid) for p in preds
    ])
    db.session.commit()
    atk = sum(1 for p in preds if p["label"] != "norm")
    return _ok(preds, batch_id=bid, total=len(preds),
               attack_count=atk, normal_count=len(preds)-atk)


@api_bp.get("/stats")
def stats():
    total    = DetectionRecord.query.count()
    # Row → dict（避免序列化问题）
    by_label = {r[0]: r[1] for r in
                db.session.query(DetectionRecord.label, func.count(DetectionRecord.id))
                .group_by(DetectionRecord.label).all()}
    by_model = {r[0]: r[1] for r in
                db.session.query(DetectionRecord.model_used, func.count(DetectionRecord.id))
                .group_by(DetectionRecord.model_used).all()}
    return _ok({"total": total, "by_label": by_label, "by_model": by_model,
                "attack": total - by_label.get("norm", 0)})


@api_bp.get("/records")
def records():
    page  = request.args.get("page",  1,  type=int)
    size  = min(request.args.get("size", 20, type=int), 100)
    label = request.args.get("label",""); model = request.args.get("model","")
    q = DetectionRecord.query.order_by(DetectionRecord.created_at.desc())
    if label: q = q.filter(DetectionRecord.label      == label)
    if model: q = q.filter(DetectionRecord.model_used == model)
    pag = q.paginate(page=page, per_page=size, error_out=False)
    return _ok([r.to_dict() for r in pag.items],
               page=page, size=size, total=pag.total, pages=pag.pages)


@api_bp.post("/compare")
def compare_models():
    """双模型对比: {"payload":"..."}"""
    b   = request.get_json(silent=True) or {}
    pay = str(b.get("payload", "")).strip()
    if not pay:
        return _err("payload 不能为空")
    if not model_manager.available:
        return _err("模型不可用，请检查云端 API 连接")
    try:
        results = {}
        for mod in ["lgbm", "textcnn"]:
            pred = model_manager.predict(mod, [pay])[0]
            results[mod] = pred
            rec = DetectionRecord(
                payload=pay, model_used=mod, label=pred["label"],
                confidence=pred["confidence"],
                all_probs=json.dumps(pred["all_probs"]),
                source="compare",
            )
            db.session.add(rec)
        db.session.commit()
        agree = results["lgbm"]["label"] == results["textcnn"]["label"]
        return _ok({"lgbm": results["lgbm"], "textcnn": results["textcnn"], "agree": agree})
    except Exception as e:
        current_app.logger.exception(e)
        return _err(f"推理失败：{e}", 500)


@api_bp.post("/analyze")
def analyze():
    """
    AI 深度分析: {"payload":"...", "label":"sqli", "label_cn":"SQL注入", "confidence":0.997}
    使用 LongCat（Anthropic 兼容接口）进行安全分析，通过 requests 直接调用，无需安装 anthropic 包。
    """
    import requests as req
    b        = request.get_json(silent=True) or {}
    pay      = str(b.get("payload", "")).strip()
    label    = b.get("label", "norm")
    label_cn = b.get("label_cn", "未知")
    conf     = float(b.get("confidence", 0)) * 100
    model_used = b.get("model", "lgbm")

    if not pay:
        return _err("payload 不能为空")

    api_key  = current_app.config.get("AI_API_KEY", "")
    api_base = current_app.config.get("AI_API_BASE", "").rstrip("/")
    ai_model = current_app.config.get("AI_MODEL", "LongCat-Flash-Lite")
    timeout  = current_app.config.get("AI_TIMEOUT", 30)
    # ai_format 在构造请求时读取，不在此处提前读

    if not api_key:
        return _err("AI_API_KEY 未配置", 500)

    mod_name = "LightGBM" if model_used == "lgbm" else "TextCNN"
    prompt = f"""你是一名专业的 Web 应用安全分析师，请对以下 HTTP 请求参数进行深度安全分析。

**待分析 Payload：**
```
{pay}
```

**ML 模型检测结果：**
- 使用模型：{mod_name}
- 检测结论：{label_cn}（{label}）
- 置信度：{conf:.1f}%

请按以下结构回答（使用 Markdown，总字数控制在 400 字以内）：

### 🔍 攻击类型判断
说明这是哪种攻击或正常流量，以及判断依据。

### ⚙️ 攻击原理
如果是攻击，简述其利用方式和危害目标；若是正常流量则说明其无害原因。

### ⚠️ 危害等级
评级为：**低 / 中 / 高 / 严重** 之一，并用一句话说明理由。

### 🛡️ 防御建议
给出 2～3 条具体可执行的防御措施（代码层面或配置层面均可）。"""

    ai_format = current_app.config.get("AI_FORMAT", "anthropic").lower()

    # 根据格式构造不同的请求体和 URL
    if ai_format == "openai":
        # OpenAI chat/completions 格式（豆包、DeepSeek、大多数国产模型）
        url  = f"{api_base}/chat/completions"
        body = {
            "model":    ai_model,
            "messages": [
                {"role": "system", "content": "你是一名专业的 Web 应用安全分析师。"},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": 1024,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
    else:
        # Anthropic messages 格式（LongCat、Claude 官方、兼容代理）
        url  = f"{api_base}/v1/messages"
        body = {
            "model":      ai_model,
            "max_tokens": 1024,
            "messages":   [{"role": "user", "content": prompt}],
        }
        headers = {
            "Authorization":     f"Bearer {api_key}",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        }

    try:
        resp = req.post(url, headers=headers, json=body, timeout=timeout)

        if not resp.ok:
            current_app.logger.error(
                f"[AI] HTTP {resp.status_code} | url={url} | body={resp.text[:300]}"
            )
            return _err(f"AI 接口返回错误：{resp.status_code}，请检查 API Key 或接口地址", 502)

        data = resp.json()
        # 自动识别响应格式，两种格式都兼容
        if "content" in data:
            # Anthropic 格式：{"content":[{"type":"text","text":"..."}]}
            text = data["content"][0]["text"]
        elif "choices" in data:
            # OpenAI 格式：{"choices":[{"message":{"content":"..."}}]}
            text = data["choices"][0]["message"]["content"]
        else:
            current_app.logger.error(f"[AI] 未知响应格式: {str(data)[:200]}")
            return _err("AI 接口返回格式未知", 502)

        return _ok({"analysis": text, "model": ai_model})

    except req.Timeout:
        return _err(f"AI 分析超时（>{timeout}s），请稍后重试", 504)
    except req.ConnectionError as e:
        current_app.logger.error(f"[AI] 连接失败: {e}")
        return _err(f"无法连接 AI 接口（{api_base}），请检查网络或接口地址", 502)
    except Exception as e:
        current_app.logger.exception(e)
        return _err(f"AI 分析失败：{e}", 500)