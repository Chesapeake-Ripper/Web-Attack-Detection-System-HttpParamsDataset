# WAD · Web Attack Detection System

> 基于机器学习的 HTTP 参数 Web 攻击多分类检测系统  
> 本科毕业设计原型 · HttpParamsDataset · Flask + LightGBM + TextCNN

---

## 📖 项目简介

WAD（Web Attack Detection）是一个针对 HTTP 请求参数值的多分类攻击检测系统。系统能够实时识别 SQL 注入、XSS 跨站脚本、命令注入、路径穿越四类攻击，并对正常流量进行区分，整体准确率超过 **99.9%**。

系统采用前后端分离的云端 API 架构：推理服务（LightGBM + TextCNN）独立部署为 FastAPI 服务，Flask Web 前端通过 HTTP 调用推理 API，两者完全解耦。

---

## ✨ 功能特性

- **🔍 单条检测** — 输入 Payload，实时返回攻击类型、置信度、各类别概率分布
- **📦 批量检测** — 支持手动输入、上传 txt/csv 文件，支持导出检测结果 CSV
- **🌐 HTTP 解析** — 粘贴 Burp Suite 格式 HTTP 报文，自动提取所有参数批量检测
- **⚖️ 双模型对比** — LightGBM 与 TextCNN 同时推理，并排展示结论差异
- **🤖 AI 深度分析** — 接入大语言模型 API，对检测结果进行攻击原理解析和防御建议
- **📋 历史记录** — SQLite 持久化，支持多维筛选、分页、单条删除
- **📊 统计看板** — 检测数量 KPI、攻击类型饼图、7 天趋势折线图、模型使用分布
- **📈 模型评估** — 展示在 HttpParamsDataset 测试集（10355 条）上的完整评测指标

---

## 🖼️ 系统截图

<img width="2372" height="2835" alt="系统截图" src="https://github.com/user-attachments/assets/89256d34-bd25-485d-b159-73a440302d91" />

---

## 🏗️ 技术架构

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

## 📊 模型性能

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

## 📁 项目结构

```
Web-Attack-Detection-System-HttpParamsDataset/
├── Train_Model/                 # 模型训练部分
│   ├── HttpParamsDataset/       # 数据集
│   ├── train.py                 # 训练脚本
│   ├── requirements.txt         # 训练依赖
│   └── outputs/                 # 训练输出（模型、图表等）
│
├── Web_Frontend/                # Web前端部分（Flask应用）
│   ├── app.py                   # Flask应用入口
│   ├── config.py                # 配置文件
│   ├── requirements.txt         # Web依赖
│   ├── templates/               # HTML模板
│   ├── static/                  # 静态资源
│   ├── blueprints/              # Flask蓝图
│   ├── inference/               # 推理引擎
│   └── models/                  # 数据模型
│
├── Inference_API/               # 后端推理部分（FastAPI服务）
│   ├── api_server.py            # FastAPI服务器
│   ├── predict.py               # 预测逻辑
│   └── outputs/                 # 模型文件
│
├── img/                         # 系统截图
└── README.md                    # 本文件
```

---

## 🚀 快速启动

### 环境要求

- Python 3.9+
- 推荐使用 Anaconda 环境

### 1. 训练模型（可选）

```bash
cd Train_Model
pip install -r requirements.txt
python train.py
```

### 2. 启动推理服务

```bash
cd Inference_API
pip install fastapi uvicorn torch scikit-learn
python api_server.py
# 服务运行在 http://localhost:9000
```

### 3. 启动Web前端

```bash
cd Web_Frontend
pip install -r requirements.txt
python app.py
# 访问 http://localhost:5000
```

---

## 📚 数据集

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

## 🔌 REST API

### Web前端API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/status` | 系统状态和模型可用性 |
| POST | `/api/detect` | 单条检测 |
| POST | `/api/detect/batch` | 批量检测（最多500条） |
| POST | `/api/compare` | 双模型对比 |
| POST | `/api/analyze` | AI 深度分析 |
| GET | `/api/stats` | 统计数据 |
| GET | `/api/records` | 历史记录（分页） |

### 推理服务API

```
Base URL: http://localhost:9000

GET  /health              → {"status": "ok"}
POST /predict             → 单条推理
POST /predict/batch       → 批量推理（推荐）
```

---

## 🚀 部署说明

### 生产环境部署

```bash
# 安装Gunicorn
pip install gunicorn

# 启动Web前端
cd Web_Frontend
gunicorn -w 2 -b 127.0.0.1:5000 "app:create_app()"

# 启动推理服务
cd Inference_API
uvicorn api_server:app --host 0.0.0.0 --port 9000 --workers 2
```

### Nginx配置示例

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static/ {
        alias /path/to/Web_Frontend/static/;
    }
}
```

---

## 📄 许可证

本项目为本科毕业设计作品，采用 [MIT License](LICENSE) 开源许可证。

---

## 🙏 致谢

- 数据集来源：CSIC2010、sqlmap、XSSYA、Vega Scanner、FuzzDB
- 技术框架：Flask、FastAPI、LightGBM、PyTorch、Bootstrap
- 指导老师：Claude

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 邮箱：1141606412@qq.com
