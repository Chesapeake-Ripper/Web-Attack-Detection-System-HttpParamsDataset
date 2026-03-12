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
