import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY              = os.environ.get("SECRET_KEY", "wad-dev-2024")
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get("DATABASE_URL") or
        "sqlite:///" + os.path.join(BASE_DIR, "instance", "wad.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # 模型目录：指向 train.py 生成的 outputs/
    MODEL_DIR         = os.environ.get("MODEL_DIR",
                            os.path.join(BASE_DIR, "outputs"))
    UPLOAD_FOLDER     = os.path.join(BASE_DIR, "uploads")
    EXPORT_FOLDER     = os.path.join(BASE_DIR, "exports")
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024   # 5 MB 上传限制
    BATCH_MAX         = 500
    PAGE_SIZE         = 20
