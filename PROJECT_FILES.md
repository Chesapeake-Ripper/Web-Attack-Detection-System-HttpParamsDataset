# WAD 项目文件清单

本文档列出项目中的所有文件，方便用户快速查找和了解项目结构。

---

## 项目根目录

| 文件 | 说明 |
|------|------|
| [README.md](README.md) | 项目主 README，包含项目概述和功能介绍 |
| [LICENSE](LICENSE) | MIT 开源许可证 |
| [CHANGELOG.md](CHANGELOG.md) | 版本更新日志 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南 |
| [SECURITY.md](SECURITY.md) | 安全政策 |
| [QUICKSTART.md](QUICKSTART.md) | 快速开始指南 |
| [Makefile](Makefile) | 常用命令集合 |
| [.gitignore](.gitignore) | Git 忽略文件配置 |
| [.gitattributes](.gitattributes) | Git 属性配置 |
| [.editorconfig](.editorconfig) | 编辑器配置 |
| [PROJECT_FILES.md](PROJECT_FILES.md) | 本文件 |
| [img/](img/) | 系统截图目录 |

---

## 模型训练模块 (Train_Model)

### 核心文件

| 文件 | 说明 |
|------|------|
| [train.py](Train_Model/train.py) | 主训练脚本 |
| [generate_paper_figures.py](Train_Model/generate_paper_figures.py) | 论文图表生成脚本 |
| [requirements.txt](Train_Model/requirements.txt) | Python 依赖列表 |
| [README.md](Train_Model/README.md) | 模块说明文档 |
| [训练结果.md](Train_Model/训练结果.md) | 训练结果记录 |
| [项目总结.md](Train_Model/项目总结.md) | 项目技术总结 |
| [Prompt.md](Train_Model/Prompt.md) | 项目需求文档 |

### 数据集 (HttpParamsDataset)

| 文件 | 说明 |
|------|------|
| [README.md](Train_Model/HttpParamsDataset/README.md) | 数据集说明 |
| [payload_train.csv](Train_Model/HttpParamsDataset/payload_train.csv) | 训练集 (20,712 条) |
| [payload_test.csv](Train_Model/HttpParamsDataset/payload_test.csv) | 测试集 (10,355 条) |
| [payload_full.csv](Train_Model/HttpParamsDataset/payload_full.csv) | 完整数据集 |
| [split_dataset.py](Train_Model/HttpParamsDataset/split_dataset.py) | 数据集划分脚本 |
| [split_log.txt](Train_Model/HttpParamsDataset/split_log.txt) | 划分操作日志 |

### 训练输出 (outputs)

| 文件 | 说明 |
|------|------|
| [lgbm_model.txt](Train_Model/outputs/lgbm_model.txt) | LightGBM 模型文件 |
| [textcnn_best.pt](Train_Model/outputs/textcnn_best.pt) | TextCNN 模型权重 |
| [char_tfidf.pkl](Train_Model/outputs/char_tfidf.pkl) | 字符级 TF-IDF 向量化器 |
| [word_tfidf.pkl](Train_Model/outputs/word_tfidf.pkl) | 词级 TF-IDF 向量化器 |
| [label_encoder.pkl](Train_Model/outputs/label_encoder.pkl) | 标签编码器 |
| [model_comparison.csv](Train_Model/outputs/model_comparison.csv) | 模型对比结果 |
| [training_logs.json](Train_Model/outputs/training_logs.json) | 训练日志 |
| [textcnn_training_log.csv](Train_Model/outputs/textcnn_training_log.csv) | TextCNN 训练日志 |
| [cm_lightgbm验证集.png](Train_Model/outputs/cm_lightgbm验证集.png) | LightGBM 验证集混淆矩阵 |
| [cm_lightgbm测试集.png](Train_Model/outputs/cm_lightgbm测试集.png) | LightGBM 测试集混淆矩阵 |
| [cm_textcnn验证集.png](Train_Model/outputs/cm_textcnn验证集.png) | TextCNN 验证集混淆矩阵 |
| [cm_textcnn测试集.png](Train_Model/outputs/cm_textcnn测试集.png) | TextCNN 测试集混淆矩阵 |
| [lgbm_feature_importance.png](Train_Model/outputs/lgbm_feature_importance.png) | 特征重要性图 |
| [textcnn_training_curves.png](Train_Model/outputs/textcnn_training_curves.png) | TextCNN 训练曲线 |
| [model_comparison.png](Train_Model/outputs/model_comparison.png) | 模型对比图 |
| [paper_figures/](Train_Model/outputs/paper_figures/) | 论文图表目录 |

---

## Web 前端模块 (Web_Frontend)

### 核心文件

| 文件 | 说明 |
|------|------|
| [app.py](Web_Frontend/app.py) | Flask 应用入口 |
| [config.py](Web_Frontend/config.py) | 配置文件 |
| [extensions.py](Web_Frontend/extensions.py) | Flask 扩展初始化 |
| [http_param_extractor.py](Web_Frontend/http_param_extractor.py) | HTTP 参数提取工具 |
| [requirements.txt](Web_Frontend/requirements.txt) | Python 依赖列表 |
| [README.md](Web_Frontend/README.md) | 模块说明文档 |

### 蓝图 (blueprints)

| 文件 | 说明 |
|------|------|
| [__init__.py](Web_Frontend/blueprints/__init__.py) | 蓝图包初始化 |
| [pages.py](Web_Frontend/blueprints/pages.py) | 页面路由 |
| [api.py](Web_Frontend/blueprints/api.py) | API 路由 |

### 推理引擎 (inference)

| 文件 | 说明 |
|------|------|
| [__init__.py](Web_Frontend/inference/__init__.py) | 推理包初始化 |
| [engine.py](Web_Frontend/inference/engine.py) | 推理引擎实现 |

### 数据模型 (models)

| 文件 | 说明 |
|------|------|
| [__init__.py](Web_Frontend/models/__init__.py) | 模型包初始化 |
| [record.py](Web_Frontend/models/record.py) | 检测记录模型 |

### 模板 (templates)

| 文件 | 说明 |
|------|------|
| [base.html](Web_Frontend/templates/base.html) | 基础模板 |
| [index.html](Web_Frontend/templates/index.html) | 首页（单条检测） |
| [batch.html](Web_Frontend/templates/batch.html) | 批量检测页面 |
| [extract.html](Web_Frontend/templates/extract.html) | HTTP 解析页面 |
| [history.html](Web_Frontend/templates/history.html) | 历史记录页面 |
| [dashboard.html](Web_Frontend/templates/dashboard.html) | 统计看板页面 |

### 静态资源 (static)

| 文件 | 说明 |
|------|------|
| [style.css](Web_Frontend/static/css/style.css) | 自定义样式 |
| [app.js](Web_Frontend/static/js/app.js) | 前端交互逻辑 |

---

## 推理服务模块 (Inference_API)

### 核心文件

| 文件 | 说明 |
|------|------|
| [api_server.py](Inference_API/api_server.py) | FastAPI 服务器入口 |
| [predict.py](Inference_API/predict.py) | 推理逻辑实现 |
| [requirements.txt](Inference_API/requirements.txt) | Python 依赖列表 |
| [README.md](Inference_API/README.md) | 模块说明文档 |
| [API.txt](Inference_API/API.txt) | API 接口说明 |
| [WAD_API_Postman文档.md](Inference_API/WAD_API_Postman文档.md) | Postman 测试文档 |
| [start.bat](Inference_API/start.bat) | Windows 启动脚本 |
| [bach_payload.txt](Inference_API/bach_payload.txt) | 批量测试数据 |

### 模型文件 (outputs)

| 文件 | 说明 |
|------|------|
| [lgbm_model.txt](Inference_API/outputs/lgbm_model.txt) | LightGBM 模型文件 |
| [textcnn_best.pt](Inference_API/outputs/textcnn_best.pt) | TextCNN 模型权重 |
| [char_tfidf.pkl](Inference_API/outputs/char_tfidf.pkl) | 字符级 TF-IDF 向量化器 |
| [word_tfidf.pkl](Inference_API/outputs/word_tfidf.pkl) | 词级 TF-IDF 向量化器 |
| [label_encoder.pkl](Inference_API/outputs/label_encoder.pkl) | 标签编码器 |

---

## 文档目录 (docs)

| 文件 | 说明 |
|------|------|
| [README.md](docs/README.md) | 文档中心索引 |
| [项目总结.md](docs/项目总结.md) | 详细技术实现和设计思路 |
| [API文档.md](docs/API文档.md) | 完整的 API 接口说明 |
| [部署文档.md](docs/部署文档.md) | 生产环境部署指南 |
| [训练结果.md](docs/训练结果.md) | 模型训练结果和性能评估 |
| [用户手册.md](docs/用户手册.md) | 系统功能使用说明 |
| [FAQ.md](docs/FAQ.md) | 常见问题解答 |

---

## GitHub 配置 (.github)

### Issue 模板

| 文件 | 说明 |
|------|------|
| [bug_report.md](.github/ISSUE_TEMPLATE/bug_report.md) | Bug 报告模板 |
| [feature_request.md](.github/ISSUE_TEMPLATE/feature_request.md) | 功能请求模板 |

### Pull Request 模板

| 文件 | 说明 |
|------|------|
| [pull_request_template.md](.github/pull_request_template.md) | PR 模板 |

### GitHub Actions

| 文件 | 说明 |
|------|------|
| [ci.yml](.github/workflows/ci.yml) | CI/CD 工作流配置 |

---

## 文件统计

| 类型 | 数量 |
|------|------|
| Markdown 文档 (.md) | 45+ |
| Python 脚本 (.py) | 20+ |
| 配置文件 | 10+ |
| 模型文件 | 10+ |
| 图片文件 | 10+ |
| **总计** | **100+** |

---

## 快速查找

### 我想了解项目

→ [README.md](README.md)

### 我想快速上手

→ [QUICKSTART.md](QUICKSTART.md)

### 我想部署项目

→ [docs/部署文档.md](docs/部署文档.md)

### 我想了解 API

→ [docs/API文档.md](docs/API文档.md)

### 我想训练模型

→ [Train_Model/README.md](Train_Model/README.md)

### 我想了解技术细节

→ [docs/项目总结.md](docs/项目总结.md)

### 我想参与贡献

→ [CONTRIBUTING.md](CONTRIBUTING.md)

### 我有问题

→ [docs/FAQ.md](docs/FAQ.md)

---

**最后更新**：2026-06-23