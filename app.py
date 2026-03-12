"""
app.py ── Flask 应用入口
运行: python app.py
"""
import os, sys

# ── 修复①：固定项目根路径注入 sys.path ────────────────────────
# 无论从哪个目录 / IDE 启动，都能正确找到本地包（inference / blueprints 等）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ──────────────────────────────────────────────────────────────

from flask import Flask
from config import Config
from extensions import db
from inference.engine import model_manager   # 包名 inference，无 PyPI 冲突

def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    model_manager.init_app(app)

    from blueprints.pages import pages_bp
    from blueprints.api   import api_bp
    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)

    with app.app_context():
        for d in [
            os.path.join(PROJECT_ROOT, "instance"),
            app.config["UPLOAD_FOLDER"],
            app.config["EXPORT_FOLDER"],
        ]:
            os.makedirs(d, exist_ok=True)
        db.create_all()
        app.logger.info("数据库初始化完成")

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=5000, debug=True)
