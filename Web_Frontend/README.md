# WAD · Web Attack Detection System

> 基于机器学习的 HTTP 参数 Web 攻击多分类检测系统  
> 本科毕业设计原型 · HttpParamsDataset · Flask + LightGBM + TextCNN

---

## 项目简介

WAD（Web Attack Detection）是一个针对 HTTP 请求参数值的多分类攻击检测系统。系统能够实时识别 SQL 注入、XSS 跨站脚本、命令注入、路径穿越四类攻击，并对正常流量进行区分，整体准确率超过 **99.9%**。

系统采用前后端分离的云端 API 架构：推理服务（LightGBM + TextCNN）独立部署为 FastAPI 服务，Flask Web 前端通过 HTTP 调用推理 API，两者完全解耦。

---

## 功能特性

- **单条检测** — 输入 Payload，实时返回攻击类型、置信度、各类别概率分布
- **批量检测** — 支持手动输入、上传 txt/csv 文件，支持导出检测结果 CSV
- **HTTP 解析** — 粘贴 Burp Suite 格式 HTTP 报文，自动提取所有参数批量检测
- **双模型对比** — LightGBM 与 TextCNN 同时推理，并排展示结论差异
- **AI 深度分析** — 接入大语言模型 API，对检测结果进行攻击原理解析和防御建议
- **历史记录** — SQLite 持久化，支持多维筛选、分页、单条删除
- **统计看板** — 检测数量 KPI、攻击类型饼图、7 天趋势折线图、模型使用分布
- **模型评估** — 展示在 HttpParamsDataset 测试集（10355 条）上的完整评测指标

---

## 技术架构

```
用户浏览器
    ↓ HTTP :80
  Nginx（反向代理 + 静态文件）
    ↓ proxy_pass
  Gunicorn + Flask（Web 前端）
    ↓ HTTP requests
  FastAPI 推理服务（云端）
    ↓
  LightGBM 模型 / TextCNN 模型
```

### 技术栈

| 层次 | 技术 |
|------|------|
| Web 框架 | Flask 3.1.0 + Flask-SQLAlchemy |
| 数据库 | SQLite（自动创建） |
| 前端 | Bootstrap 5 + Chart.js + 自定义暗色主题 |
| 推理 API | FastAPI + Uvicorn |
| 传统ML模型 | LightGBM + TF-IDF 特征（字符级 + 词级） |
| 深度学习模型 | TextCNN（PyTorch，字符级卷积） |
| 部署 | Gunicorn + Nginx + Systemd |
| AI 分析 | LongCat / 豆包 / DeepSeek（Anthropic/OpenAI 兼容接口） |

---

## 模型性能

测试集：HttpParamsDataset，共 **10,355** 条，5 分类

| 模型 | Accuracy | Precision | Recall | Macro F1 | Weighted F1 |
|------|----------|-----------|--------|----------|-------------|
| LightGBM | 99.95% | 98.18% | 99.67% | 98.88% | 99.95% |
| TextCNN | 99.94% | 99.27% | 97.99% | 98.61% | 99.94% |

### 各类别 F1-Score（LightGBM / TextCNN）

| 类别 | 测试样本数 | LightGBM F1 | TextCNN F1 |
|------|-----------|-------------|-----------|
| 正常流量 | 6434 | 1.00 | 1.00 |
| SQL注入 | 3617 | 1.00 | 1.00 |
| XSS攻击 | 177 | 1.00 | 1.00 |
| 路径穿越 | 97 | 0.99 | 1.00 |
| 命令注入 | 30 | 0.95 | 0.93 |

---

## 项目结构

```
wad_final/
├── app.py                      # Flask 应用入口（工厂函数）
├── config.py                   # 配置（API 地址、AI Key、数据库等）
├── extensions.py               # SQLAlchemy 初始化
├── requirements.txt            # 依赖（仅4个轻量包）
├── http_param_extractor.py     # HTTP 报文参数提取工具
│
├── inference/
│   ├── __init__.py
│   └── engine.py               # 云端 API 推理引擎（ModelManager）
│
├── models/
│   └── record.py               # DetectionRecord 数据模型
│
├── blueprints/
│   ├── pages.py                # 页面路由（6个页面）
│   └── api.py                  # REST API（含 AI 分析接口）
│
├── templates/
│   ├── base.html               # 导航基础模板
│   ├── index.html              # 单条检测
│   ├── batch.html              # 批量检测
│   ├── extract.html            # HTTP 报文解析
│   ├── history.html            # 历史记录
│   └── dashboard.html         # 统计看板 + 模型评估
│
└── static/
    ├── css/style.css           # 暗色网络安全主题
    └── js/app.js               # 前端交互逻辑
```

---

## 快速启动

### 环境要求

- Python 3.9+
- 云端推理 API 已运行（或本地启动 `api_server.py`）

### 安装依赖

```bash
pip install Flask==3.1.0 Flask-SQLAlchemy==3.1.1 SQLAlchemy==2.0.31 requests==2.32.3
```

### 配置

编辑 `config.py`，修改以下配置项：

```python
# 推理 API 地址
API_BASE_URL = "http://你的服务器IP:9000"

# AI 深度分析（可选）
AI_API_KEY  = "your-api-key"
AI_API_BASE = "https://api.longcat.chat/anthropic"   # 或其他兼容接口
AI_MODEL    = "LongCat-Flash-Lite"
AI_FORMAT   = "anthropic"   # "anthropic" 或 "openai"
```

### 启动

```bash
python app.py
# 访问 http://localhost:5000
```

### 生产部署（Gunicorn + Nginx）

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动
gunicorn -w 2 -b 127.0.0.1:5000 "app:create_app()"
```

---

## REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/status` | 系统状态和模型可用性 |
| POST | `/api/detect` | 单条检测 |
| POST | `/api/detect/batch` | 批量检测（最多500条） |
| POST | `/api/compare` | 双模型对比 |
| POST | `/api/analyze` | AI 深度分析 |
| GET | `/api/stats` | 统计数据 |
| GET | `/api/records` | 历史记录（分页） |

### 示例

```bash
# 单条检测
curl -X POST http://localhost:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"payload": "'\'' OR 1=1 --", "model": "lgbm"}'

# 响应
{
  "success": true,
  "data": {
    "label": "sqli",
    "label_cn": "SQL注入",
    "confidence": 0.9997,
    "icon": "🔴",
    "risk": 3,
    "all_probs": { "sqli": 0.9997, "norm": 0.0002, ... }
  }
}
```

---

## 数据集

**HttpParamsDataset** — HTTP 请求参数值多分类数据集

| 字段 | 说明 |
|------|------|
| payload | HTTP 参数值（特征） |
| length | payload 字符长度 |
| attack_type | 攻击类型标签 |
| label | 数字标签 |

| 类别 | 训练集 | 测试集 |
|------|--------|--------|
| norm（正常） | 12870 | 6434 |
| sqli（SQL注入） | 7235 | 3617 |
| xss（XSS攻击） | 355 | 177 |
| path-traversal（路径穿越） | 193 | 97 |
| cmdi（命令注入） | 59 | 30 |
| **合计** | **20712** | **10355** |

---

## 推理 API 接口（云端）

```
Base URL: http://localhost:9000

GET  /health              → {"status": "ok"}
POST /predict             → 单条推理
POST /predict/batch       → 批量推理（推荐）
```

---

## 切换 AI 分析模型

修改 `config.py` 以下四行，重启 Flask 即生效：

```python
# 豆包 Doubao-1.5-Pro
AI_API_KEY  = "your-doubao-key"
AI_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
AI_MODEL    = "doubao-1-5-pro-32k-250115"
AI_FORMAT   = "openai"

# DeepSeek-V3
AI_MODEL    = "deepseek-v3-2-251201"  # 其余同豆包

# LongCat（默认）
AI_API_BASE = "https://api.longcat.chat/anthropic"
AI_MODEL    = "LongCat-Flash-Lite"
AI_FORMAT   = "anthropic"
```

---

## License

本项目为本科毕业设计作品，仅供学习和研究使用。
