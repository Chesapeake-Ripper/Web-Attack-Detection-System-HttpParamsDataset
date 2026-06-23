# WAD 推理服务 (FastAPI)

> Web Attack Detection System - 后端推理服务  
> 提供 LightGBM 和 TextCNN 模型的推理 API

---

## 功能特性

- **单条推理** — 接收单个 payload，返回预测结果
- **批量推理** — 支持批量处理，提高效率
- **双模型支持** — 同时支持 LightGBM 和 TextCNN 模型
- **健康检查** — 提供服务状态监控接口

---

## 技术栈

- **Web框架**: FastAPI + Uvicorn
- **机器学习**: LightGBM + TF-IDF (字符级 + 词级)
- **深度学习**: TextCNN (PyTorch)
- **数据处理**: NumPy, Pandas, scikit-learn

---

## 项目结构

```
Inference_API/
├── api_server.py          # FastAPI 服务器入口
├── predict.py             # 预测逻辑实现
├── requirements.txt       # 依赖列表
├── outputs/               # 模型文件目录
│   ├── lgbm_model.txt     # LightGBM 模型
│   ├── textcnn_best.pt    # TextCNN 模型
│   ├── char_tfidf.pkl     # 字符级 TF-IDF 向量化器
│   ├── word_tfidf.pkl     # 词级 TF-IDF 向量化器
│   └── label_encoder.pkl  # 标签编码器
└── README.md              # 本文件
```

---

## 快速启动

### 环境要求

- Python 3.9+
- 推荐使用 Anaconda 环境

### 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install fastapi uvicorn torch scikit-learn lightgbm numpy pandas joblib
```

### 启动服务

```bash
# 开发模式
python api_server.py

# 或使用 uvicorn
uvicorn api_server:app --host 0.0.0.0 --port 9000 --reload

# 生产模式
uvicorn api_server:app --host 0.0.0.0 --port 9000 --workers 2
```

服务启动后访问：
- API 文档: http://localhost:9000/docs
- 健康检查: http://localhost:9000/health

---

## API 接口

### 健康检查

```http
GET /health
```

**响应示例:**
```json
{
  "status": "ok",
  "models": {
    "lightgbm": true,
    "textcnn": true
  }
}
```

### 单条推理

```http
POST /predict
Content-Type: application/json

{
  "payload": "' OR 1=1 --",
  "model": "lgbm"  // 可选: "lgbm" 或 "textcnn"
}
```

**响应示例:**
```json
{
  "success": true,
  "data": {
    "label": "sqli",
    "label_cn": "SQL注入",
    "confidence": 0.9997,
    "all_probs": {
      "norm": 0.0002,
      "sqli": 0.9997,
      "xss": 0.0001,
      "path-traversal": 0.0000,
      "cmdi": 0.0000
    }
  }
}
```

### 批量推理

```http
POST /predict/batch
Content-Type: application/json

{
  "payloads": ["' OR 1=1 --", "<script>alert(1)</script>", "normal text"],
  "model": "lgbm"
}
```

**响应示例:**
```json
{
  "success": true,
  "data": [
    {
      "payload": "' OR 1=1 --",
      "label": "sqli",
      "confidence": 0.9997
    },
    {
      "payload": "<script>alert(1)</script>",
      "label": "xss",
      "confidence": 0.9998
    },
    {
      "payload": "normal text",
      "label": "norm",
      "confidence": 0.9999
    }
  ]
}
```

---

## 模型说明

### LightGBM 模型

- **类型**: 传统机器学习模型
- **特征**: TF-IDF (字符级 + 词级)
- **优势**: 推理速度快，资源占用少
- **适用场景**: 高并发、低延迟场景

### TextCNN 模型

- **类型**: 深度学习模型
- **架构**: 字符级卷积神经网络
- **优势**: 能捕捉更复杂的特征模式
- **适用场景**: 需要更高准确率的场景

---

## 部署说明

### Systemd 服务

```ini
[Unit]
Description=WAD Inference API
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/Inference_API
ExecStart=/path/to/venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 9000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 性能优化

1. **批量处理**: 使用 `/predict/batch` 接口，减少网络开销
2. **模型选择**: 根据场景选择合适的模型
   - 高并发: LightGBM
   - 高准确率: TextCNN
3. **Worker 数量**: 根据 CPU 核心数调整 `--workers` 参数
4. **GPU 加速**: TextCNN 支持 GPU 加速（需安装 CUDA 版 PyTorch）

---

## 故障排除

### 模型加载失败

```bash
# 检查模型文件是否存在
ls -la outputs/

# 检查依赖是否完整
pip list | grep -E "torch|lightgbm|scikit-learn"
```

### 内存不足

```bash
# 减少 worker 数量
uvicorn api_server:app --host 0.0.0.0 --port 9000 --workers 1

# 或使用 LightGBM 替代 TextCNN
```

---

## 许可证

MIT License