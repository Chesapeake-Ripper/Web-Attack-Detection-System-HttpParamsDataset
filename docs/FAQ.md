# WAD 常见问题解答 (FAQ)

## 目录

- [基本问题](#基本问题)
- [功能使用](#功能使用)
- [技术问题](#技术问题)
- [部署相关](#部署相关)
- [模型相关](#模型相关)
- [故障排除](#故障排除)

---

## 基本问题

### Q: WAD 是什么？

**A:** WAD (Web Attack Detection) 是一个基于机器学习的 Web 攻击检测系统，能够实时识别 SQL 注入、XSS 跨站脚本、命令注入、路径穿越四类攻击，整体准确率超过 99.9%。

### Q: WAD 支持哪些攻击类型？

**A:** WAD 支持 5 种分类：
- **norm** - 正常流量
- **sqli** - SQL 注入
- **xss** - XSS 攻击
- **cmdi** - 命令注入
- **path-traversal** - 路径穿越

### Q: WAD 的准确率是多少？

**A:** 在 HttpParamsDataset 测试集上：
- LightGBM: 99.95% 准确率
- TextCNN: 99.94% 准确率

### Q: WAD 是免费的吗？

**A:** 是的，WAD 采用 MIT 开源许可证，完全免费使用。

### Q: WAD 可以用于商业项目吗？

**A:** 可以。MIT 许可证允许商业使用，但需要保留版权声明。

---

## 功能使用

### Q: 如何进行单条检测？

**A:** 
1. 访问系统首页
2. 在输入框中输入待检测的 Payload
3. 选择模型（LightGBM 或 TextCNN）
4. 点击"检测"按钮

### Q: 批量检测有数量限制吗？

**A:** 是的，单次批量检测最多支持 500 条 Payload。如果需要检测更多数据，可以分批进行。

### Q: 如何上传文件进行批量检测？

**A:**
1. 访问批量检测页面
2. 点击"上传文件"按钮
3. 选择 txt 或 csv 文件
4. 点击"批量检测"按钮

### Q: 支持哪些文件格式？

**A:** 支持以下格式：
- **TXT 文件**：每行一个 Payload
- **CSV 文件**：包含 `payload` 列

### Q: 如何导出检测结果？

**A:** 在批量检测结果页面，点击"导出 CSV"按钮即可下载检测结果。

### Q: HTTP 解析功能支持哪些格式？

**A:** 支持 Burp Suite 格式的 HTTP 报文，包括：
- GET 请求参数
- POST 表单数据
- Cookie
- Header

### Q: 如何使用 AI 分析功能？

**A:**
1. 先进行检测（单条或批量）
2. 在检测结果页面点击"AI 分析"按钮
3. 等待分析结果

### Q: AI 分析需要付费吗？

**A:** 默认使用 LongCat 模型，无需付费。如果需要使用其他模型（如豆包、DeepSeek），需要配置相应的 API Key。

### Q: 历史记录能保存多久？

**A:** 历史记录默认永久保存在 SQLite 数据库中。如果需要清理，可以手动删除或清空记录。

### Q: 如何筛选历史记录？

**A:** 在历史记录页面，可以使用以下筛选条件：
- 模型：LightGBM / TextCNN
- 类型：正常 / SQL注入 / XSS / 命令注入 / 路径穿越
- 时间：今天 / 本周 / 本月

---

## 技术问题

### Q: WAD 使用了什么技术栈？

**A:**
- **前端**：Bootstrap 5 + Chart.js
- **后端**：Flask + Flask-SQLAlchemy
- **推理服务**：FastAPI + Uvicorn
- **传统 ML**：LightGBM + TF-IDF
- **深度学习**：PyTorch (TextCNN)
- **数据库**：SQLite

### Q: LightGBM 和 TextCNN 有什么区别？

**A:**

| 特性 | LightGBM | TextCNN |
|------|----------|---------|
| 类型 | 传统机器学习 | 深度学习 |
| 推理速度 | 快（< 5ms） | 较慢（< 10ms） |
| 内存占用 | 少（~200MB） | 多（~500MB） |
| 准确率 | 99.95% | 99.94% |
| 适用场景 | 高并发、低延迟 | 需要更高准确率 |

### Q: 如何选择模型？

**A:**
- **高并发场景**：选择 LightGBM
- **高准确率场景**：选择 TextCNN
- **资源受限**：选择 LightGBM
- **不确定**：使用"双模型对比"功能

### Q: 系统支持哪些 Python 版本？

**A:** 推荐 Python 3.9，也支持 3.10 和 3.11。

### Q: 系统需要 GPU 吗？

**A:** 不需要。LightGBM 和 TextCNN 都可以在 CPU 上运行。如果有 GPU，TextCNN 可以使用 GPU 加速。

### Q: 系统的推理性能如何？

**A:**
- **单条推理**：LightGBM < 5ms，TextCNN < 10ms
- **批量推理（100条）**：LightGBM < 100ms，TextCNN < 200ms

---

## 部署相关

### Q: 如何部署 WAD？

**A:** 请参考 [部署文档](部署文档.md)，包含以下部署方式：
- 传统部署（开发环境）
- 生产环境部署

### Q: 系统要求是什么？

**A:**

| 环境 | CPU | 内存 | 磁盘 |
|------|-----|------|------|
| 开发环境 | 2 核+ | 4GB+ | 10GB+ |
| 测试环境 | 4 核+ | 8GB+ | 50GB+ |
| 生产环境 | 8 核+ | 16GB+ | 100GB+ |

### Q: 支持哪些操作系统？

**A:** 支持以下操作系统：
- Ubuntu 20.04/22.04 LTS
- CentOS 7/8
- Debian 10/11
- Windows 10/11
- macOS 10.15+

### Q: 如何配置 HTTPS？

**A:**
1. 安装 Certbot
2. 获取证书
3. 配置 Nginx

详细步骤请参考 [部署文档](部署文档.md)。

### Q: 如何备份数据？

**A:**
```bash
# 备份数据库
cp Web_Frontend/instance/wad.db /path/to/backup/

# 备份模型文件
cp -r Inference_API/outputs /path/to/backup/
```

---

## 模型相关

### Q: 如何训练自己的模型？

**A:**
```bash
cd Train_Model
pip install -r requirements.txt
python train.py
```

### Q: 可以使用自己的数据集吗？

**A:** 可以。准备 CSV 文件，包含 `payload` 和 `attack_type` 列，然后修改 `train.py` 中的数据加载路径。

### Q: 如何提高模型准确率？

**A:**
1. 增加训练数据
2. 调整模型超参数
3. 进行特征工程
4. 使用模型集成

### Q: 模型文件在哪里？

**A:** 模型文件位于 `Inference_API/outputs/` 目录：
- `lgbm_model.txt` - LightGBM 模型
- `textcnn_best.pt` - TextCNN 模型
- `char_tfidf.pkl` - 字符级 TF-IDF 向量化器
- `word_tfidf.pkl` - 词级 TF-IDF 向量化器
- `label_encoder.pkl` - 标签编码器

### Q: 如何更新模型？

**A:**
1. 训练新模型
2. 将新模型文件复制到 `Inference_API/outputs/`
3. 重启推理服务

---

## 故障排除

### Q: 服务无法启动怎么办？

**A:**
1. 检查日志文件
2. 检查端口占用
3. 检查依赖安装
4. 检查模型文件

```bash
# 检查端口占用
netstat -ano | findstr :5000
netstat -ano | findstr :9000

# 检查依赖
pip list | grep -E "torch|lightgbm|scikit-learn"
```

### Q: 模型加载失败怎么办？

**A:**
1. 检查模型文件是否存在
2. 检查模型文件是否完整
3. 检查依赖版本

```bash
# 检查模型文件
ls -la Inference_API/outputs/

# 重新训练模型
cd Train_Model
python train.py
```

### Q: 内存不足怎么办？

**A:**
1. 减少 Gunicorn workers 数量
2. 使用 LightGBM 替代 TextCNN
3. 增加系统内存

### Q: 检测结果不准确怎么办？

**A:**
1. 尝试使用另一个模型
2. 使用"双模型对比"功能
3. 使用"AI 深度分析"功能
4. 检查输入是否完整

### Q: 批量检测失败怎么办？

**A:**
1. 检查文件格式是否正确
2. 检查文件编码（推荐 UTF-8）
3. 检查 Payload 数量是否超过限制
4. 检查网络连接

### Q: AI 分析功能无法使用怎么办？

**A:**
1. 检查 API Key 配置
2. 检查网络连接
3. 检查 API 服务状态
4. 尝试使用其他 AI 模型

---

## 性能优化

### Q: 如何提高系统性能？

**A:**
1. **使用 Gunicorn**：多 worker 部署
2. **配置 Nginx**：反向代理和静态文件缓存
3. **使用 Redis**：缓存检测结果
4. **水平扩展**：部署多个推理服务实例

### Q: 如何减少推理延迟？

**A:**
1. 使用 LightGBM 模型
2. 减少批量检测数量
3. 使用 GPU 加速（TextCNN）
4. 部署多个推理服务实例

### Q: 如何监控系统状态？

**A:**
1. 查看日志文件
2. 使用 Supervisor 管理服务
3. 配置健康检查脚本
4. 使用监控工具（如 Prometheus）

---

## 安全相关

### Q: 系统安全吗？

**A:** 系统采用以下安全措施：
- HTTPS 加密通信
- CORS 跨域限制
- API 限流
- 输入验证

### Q: 如何配置 API 限流？

**A:**
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)
```

### Q: 如何报告安全漏洞？

**A:** 请参考 [安全政策](../SECURITY.md)，通过邮件或 GitHub Security Advisories 报告。

---

## 贡献相关

### Q: 如何参与项目开发？

**A:** 请参考 [贡献指南](../CONTRIBUTING.md)，了解：
- 代码规范
- 提交流程
- 测试要求

### Q: 如何提交 Bug 报告？

**A:**
1. 访问 GitHub Issues
2. 使用 Bug 报告模板
3. 提供详细信息

### Q: 如何建议新功能？

**A:**
1. 访问 GitHub Issues
2. 使用功能请求模板
3. 详细描述需求

---

## 联系我们

如有其他问题，请通过以下方式联系：

- **GitHub Issues**：[提交 Issue](https://github.com/your-username/Web-Attack-Detection-System-HttpParamsDataset/issues)
- **邮件**：1141606412@qq.com

---

**最后更新**：2026-06-23