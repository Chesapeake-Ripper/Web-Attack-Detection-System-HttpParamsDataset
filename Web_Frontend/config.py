"""
config.py ── Flask 应用配置
云端 API 版：不再需要 MODEL_DIR，改用 API_BASE_URL
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY              = os.environ.get("SECRET_KEY", "change-me-in-production")
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get("DATABASE_URL") or
        "sqlite:///" + os.path.join(BASE_DIR, "instance", "wad.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ── 云端推理 API ──────────────────────────────────────────
    # 优先读取环境变量，方便部署时切换；默认指向自建服务器
    API_BASE_URL  = os.environ.get("API_BASE_URL", "http://localhost:8000")
    API_TIMEOUT   = int(os.environ.get("API_TIMEOUT", "15"))

    # ── 文件上传 / 导出 ───────────────────────────────────────
    UPLOAD_FOLDER      = os.path.join(BASE_DIR, "uploads")
    EXPORT_FOLDER      = os.path.join(BASE_DIR, "exports")
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024   # 5 MB

    # ── AI 深度分析 ──────────────────────────────────────────
    # 切换模型只需修改下面三行，重启 Flask 即生效
    # 请在环境变量中设置 AI_API_KEY，或在此处填入你的 API Key
    AI_API_KEY  = os.environ.get("AI_API_KEY", "your-api-key-here")
    AI_API_BASE = os.environ.get("AI_API_BASE", "https://api.longcat.chat/anthropic")
    AI_MODEL    = os.environ.get("AI_MODEL", "LongCat-Flash-Lite")
    AI_FORMAT   = os.environ.get("AI_FORMAT", "anthropic")

    # ── 分页 / 批量 ───────────────────────────────────────────
    BATCH_MAX = 500
    PAGE_SIZE = 20