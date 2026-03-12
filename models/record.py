from __future__ import annotations
import json
from datetime import datetime
from extensions import db


class DetectionRecord(db.Model):
    __tablename__ = "detection_records"

    id         = db.Column(db.Integer,    primary_key=True)
    payload    = db.Column(db.Text,       nullable=False)
    model_used = db.Column(db.String(32), nullable=False)    # lgbm / textcnn
    label      = db.Column(db.String(32), nullable=False)
    confidence = db.Column(db.Float,      nullable=False)
    all_probs  = db.Column(db.Text,       nullable=True)     # JSON 字符串
    source     = db.Column(db.String(16), default="single")  # single/batch/api
    batch_id   = db.Column(db.String(64), nullable=True, index=True)
    created_at = db.Column(db.DateTime,   default=datetime.utcnow, index=True)

    _META = {
        "norm"          : ("正常流量", "bdg-norm"),
        "sqli"          : ("SQL注入",  "bdg-sqli"),
        "xss"           : ("XSS攻击",  "bdg-xss"),
        "cmdi"          : ("命令注入", "bdg-cmdi"),
        "path-traversal": ("路径穿越", "bdg-path"),
    }

    @property
    def label_cn(self)  -> str:  return self._META.get(self.label, (self.label,))[0]
    @property
    def label_cls(self) -> str:  return self._META.get(self.label, ("", "bdg-norm"))[1]
    @property
    def is_attack(self) -> bool: return self.label != "norm"

    def to_dict(self) -> dict:
        return {
            "id"        : self.id,
            "payload"   : self.payload,
            "model"     : self.model_used,
            "label"     : self.label,
            "label_cn"  : self.label_cn,
            "confidence": round(self.confidence * 100, 2),
            "all_probs" : json.loads(self.all_probs) if self.all_probs else {},
            "source"    : self.source,
            "batch_id"  : self.batch_id,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
