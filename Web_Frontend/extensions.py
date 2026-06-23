# !/user/bin/env python3
# -*- coding: utf-8 -*-

from flask_sqlalchemy import SQLAlchemy

# 全局单例，被 app.py 和 models/ 共同 import
db = SQLAlchemy()
