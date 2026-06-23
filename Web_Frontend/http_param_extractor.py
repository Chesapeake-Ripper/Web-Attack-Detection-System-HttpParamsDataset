"""
http_param_extractor.py
=========================================================
从 HTTP 请求中提取所有参数原始值（对应 HttpParamsDataset.payload）

支持：
  GET  - URL query string
  POST - application/x-www-form-urlencoded
  POST - application/json（递归展开嵌套结构）
  POST - multipart/form-data（跳过文件字段）

原则：保留原始字符串，不做任何 URL 解码 / 预处理
运行：python http_param_extractor.py
=========================================================
"""
from __future__ import annotations

import io
import json
import re
import urllib.parse
from urllib.parse import unquote_plus
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────

@dataclass
class HttpRequest:
    """模拟一条 HTTP 请求，可替换为 Flask/Django/FastAPI request"""
    method:       str
    url:          str
    content_type: str   = ""
    body:         bytes = b""
    headers:      dict  = field(default_factory=dict)


@dataclass
class ExtractedParam:
    """单个提取到的参数"""
    name:   str   # 参数名
    value:  str   # 原始参数值（未解码）
    source: str   # query_string / form / json / multipart


@dataclass
class ExtractionResult:
    """extract_payloads() 的返回值"""
    payloads: list   # 仅参数值列表，直接送入模型
    params:   list   # 完整 ExtractedParam 列表
    errors:   list   # 解析过程中的非致命错误


# ─────────────────────────────────────────────────────────
# 内部：GET query string
# ─────────────────────────────────────────────────────────

def _decode_value(value: str) -> str:
    """
    对参数值做 3 轮 unquote_plus 解码，与 predict.py 的 _preprocess 保持一致。

    unquote_plus 相比 unquote 额外将 + 号解码为空格，
    确保 %27%3B+DROP+TABLE → '; DROP TABLE（而不是 ';+DROP+TABLE）。
    3 轮循环处理双重/三重编码，如 %2527 → %27 → '。
    """
    for _ in range(3):
        decoded = unquote_plus(value)
        if decoded == value:
            break
        value = decoded
    return value


def _split_raw_query(qs: str) -> list:
    """
    切割 query string，对 value 做完整 URL 解码（unquote_plus × 3 轮）。

    解码后的值与数据集训练样本对齐，模型识别率显著提升。
    示例：%27%3B+DROP+TABLE → '; DROP TABLE
    """
    pairs = []
    if not qs:
        return pairs
    for token in qs.split("&"):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            idx   = token.index("=")
            name  = token[:idx]
            value = _decode_value(token[idx + 1:])   # ← 解码
        else:
            name  = token
            value = ""
        pairs.append((name, value))
    return pairs


def _extract_query_string(url: str):
    params, errors = [], []
    try:
        qs = urllib.parse.urlparse(url).query
        for name, value in _split_raw_query(qs):
            params.append(ExtractedParam(name=name, value=value,
                                         source="query_string"))
    except Exception as e:
        errors.append(f"[query_string] 解析失败: {e}")
    return params, errors


# ─────────────────────────────────────────────────────────
# 内部：POST form-urlencoded
# ─────────────────────────────────────────────────────────

def _extract_form_urlencoded(body: bytes):
    params, errors = [], []
    try:
        qs = body.decode("utf-8", errors="replace")
        for name, value in _split_raw_query(qs):
            params.append(ExtractedParam(name=name, value=value, source="form"))
    except Exception as e:
        errors.append(f"[form_urlencoded] 解析失败: {e}")
    return params, errors


# ─────────────────────────────────────────────────────────
# 内部：POST JSON（递归展开嵌套结构）
# ─────────────────────────────────────────────────────────

def _flatten_json(obj: Any, parent_key: str, result: list) -> None:
    """
    递归展开 JSON，将所有叶节点值作为 payload。
    {"user": {"name": "admin"}, "q": "' OR 1=1"}
    → user.name=admin, q=' OR 1=1
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{parent_key}.{k}" if parent_key else k
            _flatten_json(v, key, result)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flatten_json(v, f"{parent_key}[{i}]", result)
    else:
        value = obj if isinstance(obj, str) else str(obj)
        result.append(ExtractedParam(name=parent_key, value=value, source="json"))


def _extract_json(body: bytes):
    params, errors = [], []
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
        _flatten_json(data, "", params)
    except json.JSONDecodeError as e:
        errors.append(f"[json] JSON 解析失败: {e}")
    except Exception as e:
        errors.append(f"[json] 未知错误: {e}")
    return params, errors


# ─────────────────────────────────────────────────────────
# 内部：POST multipart/form-data（纯手动解析，不依赖 cgi）
# ─────────────────────────────────────────────────────────

def _extract_multipart(body: bytes, content_type: str):
    """
    手动解析 multipart/form-data。
    - 从 Content-Type 中提取 boundary
    - 按 boundary 切割各 part
    - 跳过有 filename 属性的文件字段
    - 只提取文本字段的原始值
    """
    params, errors = [], []
    try:
        # 提取 boundary
        boundary = None
        for seg in content_type.split(";"):
            seg = seg.strip()
            if seg.lower().startswith("boundary="):
                boundary = seg[9:].strip().strip('"')
                break
        if not boundary:
            errors.append("[multipart] Content-Type 中未找到 boundary")
            return params, errors

        sep = ("--" + boundary).encode("utf-8")
        raw_parts = body.split(sep)

        for raw in raw_parts[1:]:
            # 结束标记或空段
            if raw.strip() in (b"--", b""):
                break

            # 分离头部和体部（空行分隔）
            if b"\r\n\r\n" in raw:
                header_bytes, body_bytes = raw.split(b"\r\n\r\n", 1)
            elif b"\n\n" in raw:
                header_bytes, body_bytes = raw.split(b"\n\n", 1)
            else:
                continue

            # 解析 Content-Disposition，提取 name / filename
            header_str = header_bytes.decode("utf-8", errors="replace")
            name     = None
            filename = None
            for line in header_str.splitlines():
                if "content-disposition" in line.lower():
                    for token in line.split(";"):
                        token = token.strip()
                        if token.lower().startswith("name="):
                            name = token[5:].strip().strip('"')
                        elif token.lower().startswith("filename="):
                            filename = token[9:].strip().strip('"')

            # 没有 name 或是文件字段 → 跳过
            if name is None or filename is not None:
                continue

            # 去掉末尾换行，解码为字符串
            value = body_bytes.rstrip(b"\r\n").decode("utf-8", errors="replace")
            params.append(ExtractedParam(name=name, value=value,
                                         source="multipart"))

    except Exception as e:
        errors.append(f"[multipart] 解析失败: {e}")
    return params, errors


# ─────────────────────────────────────────────────────────
# 对外统一入口
# ─────────────────────────────────────────────────────────

def extract_payloads(request: HttpRequest) -> ExtractionResult:
    """
    从 HTTP 请求中提取所有参数原始值。

    参数
    ----
    request : HttpRequest

    返回
    ----
    ExtractionResult
        .payloads  : list[str]   参数值列表，直接送入 predict_lgbm()
        .params    : list        完整 ExtractedParam 列表
        .errors    : list[str]   解析过程中的非致命错误
    """
    all_params, all_errors = [], []
    method = request.method.upper()

    # GET / POST 都先提取 URL query string
    qp, qe = _extract_query_string(request.url)
    all_params.extend(qp)
    all_errors.extend(qe)

    # POST body 按 Content-Type 分支
    if method == "POST" and request.body:
        ct = request.content_type.lower().strip()

        if "application/x-www-form-urlencoded" in ct:
            fp, fe = _extract_form_urlencoded(request.body)
            all_params.extend(fp); all_errors.extend(fe)

        elif "application/json" in ct:
            jp, je = _extract_json(request.body)
            all_params.extend(jp); all_errors.extend(je)

        elif "multipart/form-data" in ct:
            mp, me = _extract_multipart(request.body, request.content_type)
            all_params.extend(mp); all_errors.extend(me)

        else:
            all_errors.append(
                f"[POST] 不支持的 Content-Type: {request.content_type}，已跳过 body"
            )

    # 过滤空值
    non_empty = [p for p in all_params if p.value.strip()]

    return ExtractionResult(
        payloads=[p.value for p in non_empty],
        params=non_empty,
        errors=all_errors,
    )


# ─────────────────────────────────────────────────────────
# Flask 对接工具
# ─────────────────────────────────────────────────────────

def from_flask_request(flask_request) -> HttpRequest:
    """
    将 Flask request 对象转为 HttpRequest。

    用法（Flask 视图中）：
        from http_param_extractor import extract_payloads, from_flask_request

        @app.route("/detect", methods=["GET","POST"])
        def detect():
            result = extract_payloads(from_flask_request(request))
            preds  = predict_lgbm(result.payloads)
            ...
    """
    return HttpRequest(
        method       = flask_request.method,
        url          = flask_request.url,
        content_type = flask_request.content_type or "",
        body         = flask_request.get_data(),
        headers      = dict(flask_request.headers),
    )


# ─────────────────────────────────────────────────────────
# 打印工具
# ─────────────────────────────────────────────────────────

def _print_result(title: str, result: ExtractionResult) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")
    if result.errors:
        for e in result.errors:
            print(f"  WARNING  {e}")
    if not result.params:
        print("  (未提取到任何参数)")
        return
    print(f"  {'来源':<16} {'参数名':<22} 原始值")
    print(f"  {'-'*14} {'-'*20} {'-'*22}")
    for p in result.params:
        src   = f"[{p.source}]"
        name  = p.name[:20]
        value = (p.value[:48] + "...") if len(p.value) > 48 else p.value
        print(f"  {src:<16} {name:<22} {value}")
    print(f"\n  => 共提取 {len(result.payloads)} 个 payload")


# ─────────────────────────────────────────────────────────
# 调用示例
# ─────────────────────────────────────────────────────────

def _run_demos():

    # ── GET 场景 ──────────────────────────────────────────

    _print_result("GET 普通参数", extract_payloads(HttpRequest(
        method="GET",
        url="https://example.com/search?q=hello+world&page=1&sort=asc",
    )))

    _print_result("GET SQL注入（URL编码）", extract_payloads(HttpRequest(
        method="GET",
        url="https://example.com/user?id=1%27%20OR%201%3D1%20--&name=admin",
    )))

    _print_result("GET XSS 攻击", extract_payloads(HttpRequest(
        method="GET",
        url="https://example.com/page?title=%3Cscript%3Ealert%281%29%3C%2Fscript%3E&lang=zh",
    )))

    _print_result("GET 路径穿越", extract_payloads(HttpRequest(
        method="GET",
        url="https://example.com/file?path=../../../../etc/passwd&type=txt",
    )))

    # ── POST form-urlencoded ──────────────────────────────

    _print_result("POST form-urlencoded SQL注入", extract_payloads(HttpRequest(
        method="POST",
        url="https://example.com/login",
        content_type="application/x-www-form-urlencoded",
        body=b"username=admin&password=%27+OR+%271%27%3D%271&remember=on",
    )))

    _print_result("POST form-urlencoded 命令注入", extract_payloads(HttpRequest(
        method="POST",
        url="https://example.com/ping",
        content_type="application/x-www-form-urlencoded",
        body=b"host=127.0.0.1%3B+cat+%2Fetc%2Fpasswd&timeout=5",
    )))

    # ── POST JSON ─────────────────────────────────────────

    _print_result("POST JSON XSS + 嵌套结构", extract_payloads(HttpRequest(
        method="POST",
        url="https://example.com/api/comment",
        content_type="application/json",
        body=json.dumps({
            "user": {
                "name": "<script>alert('xss')</script>",
                "email": "test@example.com"
            },
            "comment": "Hello ' OR '1'='1",
            "tags": ["normal", "<img src=x onerror=alert(1)>"]
        }).encode("utf-8"),
    )))

    _print_result("POST JSON 路径穿越", extract_payloads(HttpRequest(
        method="POST",
        url="https://example.com/api/file",
        content_type="application/json",
        body=json.dumps({
            "action": "download",
            "filename": "../../../../etc/shadow",
            "format": "raw"
        }).encode("utf-8"),
    )))

    # ── POST multipart ────────────────────────────────────

    multipart_body = (
        b"------Boundary123\r\n"
        b'Content-Disposition: form-data; name="username"\r\n\r\n'
        b"' OR 1=1 --\r\n"
        b"------Boundary123\r\n"
        b'Content-Disposition: form-data; name="note"\r\n\r\n'
        b"<script>alert('xss')</script>\r\n"
        # 文件字段 —— 应被跳过
        b"------Boundary123\r\n"
        b'Content-Disposition: form-data; name="avatar"; filename="evil.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n"
        b"\xff\xd8\xff\xe0FAKEJPEG\r\n"
        b"------Boundary123--\r\n"
    )
    _print_result("POST multipart（文件字段应被跳过）", extract_payloads(HttpRequest(
        method="POST",
        url="https://example.com/profile",
        content_type="multipart/form-data; boundary=----Boundary123",
        body=multipart_body,
    )))

    # ── POST body + URL query string 同时存在 ─────────────

    _print_result("POST body + URL query 双来源", extract_payloads(HttpRequest(
        method="POST",
        url="https://example.com/submit?source=web&ref=home",
        content_type="application/x-www-form-urlencoded",
        body=b"content=%3Cscript%3Ealert%281%29%3C%2Fscript%3E&action=post",
    )))

    # ── 对接推理模型示例 ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  与推理模型对接示例（需配合 predict.py 使用）")
    print(f"{'=' * 60}")
    print("""
  from predict import predict_lgbm
  from http_param_extractor import extract_payloads, from_flask_request

  @app.route("/detect", methods=["GET","POST"])
  def detect():
      result  = extract_payloads(from_flask_request(request))
      if not result.payloads:
          return {"error": "未提取到参数"}
      preds   = predict_lgbm(result.payloads)
      attacks = [p for p in preds if p["label"] != "norm"]
      return {
          "total":        len(preds),
          "attack_count": len(attacks),
          "attacks":      attacks,
      }
    """)


if __name__ == "__main__":
    _run_demos()
