"""
blueprints/pages.py ── 页面路由
修复⑤：SQLAlchemy Row 对象 → 普通 Python 列表（避免 tojson 序列化报错）
"""
from __future__ import annotations

import csv, io, json, uuid
from datetime import datetime, timedelta

from flask import (Blueprint, render_template, request,
                   redirect, url_for, flash, current_app, Response)
from sqlalchemy import func

from extensions import db
from models.record import DetectionRecord
from inference.engine import model_manager
from http_param_extractor import extract_payloads, HttpRequest

pages_bp = Blueprint("pages", __name__)

_CN = {
    "norm":"正常流量","sqli":"SQL注入","xss":"XSS攻击",
    "cmdi":"命令注入","path-traversal":"路径穿越",
}


# ── 单条检测 ───────────────────────────────────────────────────
@pages_bp.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        payload = request.form.get("payload", "").strip()
        model   = request.form.get("model", "lgbm")
        if not payload:
            flash("请输入 Payload 内容", "warning")
            return redirect(url_for("pages.index"))
        try:
            pred = model_manager.predict(model, [payload])[0]
            rec  = DetectionRecord(
                payload=payload, model_used=model,
                label=pred["label"], confidence=pred["confidence"],
                all_probs=json.dumps(pred["all_probs"]), source="single",
            )
            db.session.add(rec); db.session.commit()
            result = {**pred, "record_id": rec.id}
        except Exception as e:
            flash(f"检测失败：{e}", "danger")

    return render_template("index.html", result=result,
                           available=model_manager.available)


# ── 批量检测（手动输入）────────────────────────────────────────
@pages_bp.route("/batch", methods=["GET", "POST"])
def batch():
    results = stats = batch_id = None
    if request.method == "POST":
        model   = request.form.get("model", "lgbm")
        raw     = request.form.get("payloads", "").strip()
        max_n   = current_app.config["BATCH_MAX"]
        lines   = [l.strip() for l in raw.splitlines() if l.strip()]
        if not lines:
            flash("请输入至少一条 Payload", "warning")
            return redirect(url_for("pages.batch"))
        if len(lines) > max_n:
            flash(f"超出 {max_n} 条限制，已截断", "warning")
            lines = lines[:max_n]
        try:
            preds    = model_manager.predict(model, lines)
            batch_id = _save_batch(preds, model, "batch")
            results, stats = preds, _stats(preds)
        except Exception as e:
            flash(f"检测失败：{e}", "danger")
    return render_template("batch.html", results=results, stats=stats,
                           batch_id=batch_id, available=model_manager.available)


# ── 批量检测（文件上传）────────────────────────────────────────
@pages_bp.route("/batch/upload", methods=["POST"])
def batch_upload():
    model = request.form.get("model", "lgbm")
    file  = request.files.get("file")
    if not file or not file.filename:
        flash("请选择文件", "warning"); return redirect(url_for("pages.batch"))

    fname = file.filename.lower()
    if not (fname.endswith(".txt") or fname.endswith(".csv")):
        flash("仅支持 .txt 或 .csv 文件", "warning")
        return redirect(url_for("pages.batch"))

    try:
        content = file.read().decode("utf-8", errors="ignore")
        if fname.endswith(".csv"):
            reader = csv.DictReader(io.StringIO(content))
            fields = reader.fieldnames or []
            pcol   = "payload" if "payload" in fields else (fields[0] if fields else None)
            lines  = ([row[pcol].strip() for row in reader if row.get(pcol,"").strip()]
                      if pcol else
                      [l.strip() for l in content.splitlines() if l.strip()])
        else:
            lines = [l.strip() for l in content.splitlines() if l.strip()]

        max_n = current_app.config["BATCH_MAX"]
        if len(lines) > max_n:
            flash(f"文件超出 {max_n} 条限制，已截断", "warning")
            lines = lines[:max_n]
        if not lines:
            flash("文件中未检测到有效内容", "warning")
            return redirect(url_for("pages.batch"))

        preds    = model_manager.predict(model, lines)
        batch_id = _save_batch(preds, model, "batch")
        flash(f"上传成功，共检测 {len(preds)} 条", "success")
        return render_template("batch.html", results=preds, stats=_stats(preds),
                               batch_id=batch_id, available=model_manager.available)
    except Exception as e:
        flash(f"文件处理失败：{e}", "danger")
        return redirect(url_for("pages.batch"))


# ── 导出 CSV ───────────────────────────────────────────────────
@pages_bp.route("/export/<bid>")
def export_batch(bid: str):
    recs = (DetectionRecord.query.filter_by(batch_id=bid)
            .order_by(DetectionRecord.id).all())
    if not recs:
        flash("未找到该批次记录", "warning")
        return redirect(url_for("pages.history"))

    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["id","payload","model","label","label_cn","confidence_%","created_at"])
    for r in recs:
        w.writerow([r.id, r.payload, r.model_used, r.label, r.label_cn,
                    round(r.confidence*100, 2),
                    r.created_at.strftime("%Y-%m-%d %H:%M:%S")])
    buf.seek(0)
    # utf-8-sig：Windows Excel 正确识别中文
    return Response(
        buf.getvalue().encode("utf-8-sig"),
        mimetype="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="wad_export_{bid[:8]}.csv"'}
    )


# ── 历史记录 ───────────────────────────────────────────────────
@pages_bp.route("/history")
def history():
    page  = request.args.get("page",    1,  type=int)
    fl    = request.args.get("label",  "")
    fm    = request.args.get("model",  "")
    fs    = request.args.get("source", "")
    fk    = request.args.get("keyword","")

    q = DetectionRecord.query.order_by(DetectionRecord.created_at.desc())
    if fl: q = q.filter(DetectionRecord.label      == fl)
    if fm: q = q.filter(DetectionRecord.model_used == fm)
    if fs: q = q.filter(DetectionRecord.source     == fs)
    if fk: q = q.filter(DetectionRecord.payload.contains(fk))

    pag = q.paginate(page=page,
                     per_page=current_app.config["PAGE_SIZE"],
                     error_out=False)
    return render_template("history.html", pagination=pag,
                           f_label=fl, f_model=fm, f_source=fs, f_kw=fk,
                           available=model_manager.available)


# ── 统计看板 ───────────────────────────────────────────────────
@pages_bp.route("/dashboard")
def dashboard():
    total  = DetectionRecord.query.count()
    attack = DetectionRecord.query.filter(DetectionRecord.label != "norm").count()

    # 修复⑤：.all() 返回 SQLAlchemy Row，tojson 无法序列化
    # 用列表推导式转成普通 Python 列表 [[label, count], ...]
    by_label = [
        [r[0], r[1]] for r in
        db.session.query(DetectionRecord.label, func.count(DetectionRecord.id))
        .group_by(DetectionRecord.label).all()
    ]
    by_model = [
        [r[0], r[1]] for r in
        db.session.query(DetectionRecord.model_used, func.count(DetectionRecord.id))
        .group_by(DetectionRecord.model_used).all()
    ]

    # 近 7 天趋势
    trend = []
    for i in range(6, -1, -1):
        day   = datetime.utcnow().date() - timedelta(days=i)
        s     = datetime.combine(day, datetime.min.time())
        e     = s + timedelta(days=1)
        t = DetectionRecord.query.filter(
            DetectionRecord.created_at >= s, DetectionRecord.created_at < e).count()
        a = DetectionRecord.query.filter(
            DetectionRecord.created_at >= s, DetectionRecord.created_at < e,
            DetectionRecord.label != "norm").count()
        trend.append({"date": day.strftime("%m/%d"), "total": t, "attack": a})

    recent = (DetectionRecord.query
              .order_by(DetectionRecord.created_at.desc()).limit(10).all())

    return render_template("dashboard.html",
                           total=total, attack=attack, normal=total-attack,
                           attack_rate=round(attack/total*100,1) if total else 0,
                           by_label=by_label, by_model=by_model,
                           trend=trend, recent=recent)


# ── HTTP 请求解析 + 批量检测 ──────────────────────────────────────
@pages_bp.route("/extract", methods=["GET", "POST"])
def extract():
    """
    粘贴原始 HTTP 请求报文 → 自动提取所有参数 → 批量检测。
    支持 GET query string / POST form / JSON / multipart。
    """
    results = stats = batch_id = None
    params_detail = []          # 带来源信息的详细列表，供模板展示
    raw_request = ""            # 回显用户输入
    parse_errors = []

    if request.method == "POST":
        raw_request = request.form.get("raw_request", "").strip()
        model       = request.form.get("model", "lgbm")

        if not raw_request:
            flash("请粘贴 HTTP 请求报文内容", "warning")
        else:
            try:
                http_req = _parse_raw_http(raw_request)
                extraction = extract_payloads(http_req)
                parse_errors = extraction.errors
                params_detail = [
                    {"name": p.name, "value": p.value, "source": p.source}
                    for p in extraction.params
                ]

                if not extraction.payloads:
                    flash("未从请求中提取到有效参数，请检查报文格式", "warning")
                else:
                    max_n = current_app.config["BATCH_MAX"]
                    payloads = extraction.payloads[:max_n]
                    if len(extraction.payloads) > max_n:
                        flash(f"参数数量超出 {max_n} 条限制，已截断", "warning")

                    preds    = model_manager.predict(model, payloads)
                    batch_id = _save_batch(preds, model, "extract")
                    results, stats = preds, _stats(preds)

                    # 把参数来源信息合并进 preds 供模板使用
                    src_map = {p.value: p.source for p in extraction.params}
                    for pred in results:
                        pred["param_source"] = src_map.get(pred["payload"], "")

            except Exception as e:
                flash(f"解析失败：{e}", "danger")

    return render_template(
        "extract.html",
        results=results, stats=stats, batch_id=batch_id,
        params_detail=params_detail, parse_errors=parse_errors,
        raw_request=raw_request,
        available=model_manager.available,
    )


def _parse_raw_http(raw: str) -> HttpRequest:
    """
    将用户粘贴的原始 HTTP 请求报文解析为 HttpRequest 对象。

    支持格式（Burp Suite / DevTools / curl -v 复制）：
        GET /path?a=1&b=2 HTTP/1.1
        Host: example.com
        Content-Type: application/json

        {"key": "value"}
    """
    lines = raw.replace("\r\n", "\n").splitlines()
    if not lines:
        raise ValueError("请求报文为空")

    # ── 解析请求行 ──────────────────────────────────────────
    request_line = lines[0].strip()
    parts = request_line.split()
    if len(parts) < 2:
        raise ValueError(f"无法解析请求行：{request_line}")

    method = parts[0].upper()
    path   = parts[1]          # 可能含 query string

    # ── 解析头部 & 定位空行 ────────────────────────────────
    headers = {}
    body_start = len(lines)    # 默认无 body

    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "":
            body_start = i + 1
            break
        if ":" in line:
            k, _, v = line.partition(":")
            headers[k.strip().lower()] = v.strip()

    # ── 拼出完整 URL ───────────────────────────────────────
    host = headers.get("host", "example.com")
    if path.startswith("http://") or path.startswith("https://"):
        url = path
    else:
        scheme = "https" if headers.get("x-forwarded-proto") == "https" else "http"
        url = f"{scheme}://{host}{path}"

    # ── 提取 body ──────────────────────────────────────────
    body_lines = lines[body_start:] if body_start < len(lines) else []
    body = "\n".join(body_lines).strip().encode("utf-8")

    content_type = headers.get("content-type", "")

    return HttpRequest(
        method=method,
        url=url,
        content_type=content_type,
        body=body,
        headers=headers,
    )


# ── 删除记录 ───────────────────────────────────────────────────
@pages_bp.post("/record/<int:rid>/delete")
def delete_record(rid: int):
    rec = DetectionRecord.query.get_or_404(rid)
    db.session.delete(rec); db.session.commit()
    flash("记录已删除", "success")
    return redirect(request.referrer or url_for("pages.history"))


# ── 内部工具 ───────────────────────────────────────────────────
def _save_batch(preds, model_key, source) -> str:
    bid = str(uuid.uuid4())
    db.session.bulk_save_objects([
        DetectionRecord(
            payload=p["payload"], model_used=model_key,
            label=p["label"], confidence=p["confidence"],
            all_probs=json.dumps(p["all_probs"]),
            source=source, batch_id=bid,
        ) for p in preds
    ])
    db.session.commit()
    return bid


def _stats(preds) -> dict:
    cnt = {}
    for p in preds: cnt[p["label"]] = cnt.get(p["label"], 0) + 1
    return {"total": len(preds), "normal": cnt.get("norm", 0),
            "attack": len(preds)-cnt.get("norm",0), "by_label": cnt}
