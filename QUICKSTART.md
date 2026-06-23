# 快速开始指南

本指南帮助你在 5 分钟内快速启动 WAD (Web Attack Detection System) 项目。

---

## 前置条件

- Python 3.9+
- Git
- 推荐使用 Anaconda 或 Miniconda

---

## 1. 克隆项目

```bash
git clone https://github.com/your-username/Web-Attack-Detection-System-HttpParamsDataset.git
cd Web-Attack-Detection-System-HttpParamsDataset
```

---

## 2. 环境配置

### 方式一：使用 Conda（推荐）

```bash
# 创建虚拟环境
conda create -n wad python=3.9
conda activate wad

# 安装 PyTorch（CPU版本，如果不需要GPU）
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 或者安装 GPU 版本（需要 CUDA）
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 方式二：使用 venv

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

---

## 3. 启动推理服务

推理服务是系统的核心，提供模型推理 API。

```bash
# 进入推理服务目录
cd Inference_API

# 安装依赖
pip install -r requirements.txt

# 启动服务
python api_server.py
```

**验证服务启动：**

```bash
# 访问健康检查接口
curl http://localhost:9000/health

# 预期响应：
# {"status": "ok", "models": {"lightgbm": true, "textcnn": true}}
```

**保持此终端运行，不要关闭。**

---

## 4. 启动 Web 前端

打开新的终端窗口：

```bash
# 激活虚拟环境（如果使用conda）
conda activate wad

# 进入Web前端目录
cd Web_Frontend

# 安装依赖
pip install -r requirements.txt

# 启动Flask应用
python app.py
```

**访问应用：**

打开浏览器访问：http://localhost:5000

---

## 5. 测试功能

### 单条检测

1. 访问首页 http://localhost:5000
2. 在输入框中输入测试 payload：
   ```
   ' OR 1=1 --
   ```
3. 点击"检测"按钮
4. 查看检测结果

### 批量检测

1. 访问批量检测页面 http://localhost:5000/batch
2. 输入多个 payload（每行一个）：
   ```
   ' OR 1=1 --
   <script>alert(1)</script>
   ../../../etc/passwd
   normal text
   ```
3. 点击"批量检测"按钮
4. 查看检测结果，可导出 CSV

### HTTP 报文解析

1. 访问 HTTP 解析页面 http://localhost:5000/extract
2. 粘贴 Burp Suite 格式的 HTTP 报文：
   ```
   GET /search?q=test&id=1 HTTP/1.1
   Host: example.com
   User-Agent: Mozilla/5.0
   ```
3. 点击"解析"按钮
4. 查看解析结果并批量检测

---

## 6. 训练自己的模型（可选）

如果你想重新训练模型或使用自己的数据集：

```bash
# 进入训练目录
cd Train_Model

# 安装依赖
pip install -r requirements.txt

# 运行训练脚本
python train.py
```

**训练完成后：**
- 模型文件保存在 `outputs/` 目录
- 将新的模型文件复制到 `Inference_API/outputs/` 目录
- 重启推理服务

---

## 常见问题

### Q1: 启动推理服务时提示 "ModuleNotFoundError"

**解决方案：**
```bash
pip install fastapi uvicorn torch scikit-learn lightgbm numpy pandas joblib
```

### Q2: 启动 Web 前端时提示 "Connection refused"

**原因：** 推理服务未启动或地址配置错误

**解决方案：**
1. 确保推理服务正在运行
2. 检查 `Web_Frontend/config.py` 中的 `API_BASE_URL` 配置
3. 默认地址：`http://localhost:9000`

### Q3: 模型加载失败

**解决方案：**
```bash
# 检查模型文件是否存在
ls -la Inference_API/outputs/

# 应该包含以下文件：
# - lgbm_model.txt
# - textcnn_best.pt
# - char_tfidf.pkl
# - word_tfidf.pkl
# - label_encoder.pkl
```

### Q4: 内存不足

**解决方案：**
```bash
# 使用 LightGBM 替代 TextCNN（资源占用更少）
# 在检测时选择 "LightGBM" 模型

# 或者减少批量检测的数量
```

### Q5: 端口被占用

**解决方案：**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <进程ID> /F

# Linux/macOS
lsof -i :5000
kill -9 <进程ID>
```

---

## 配置说明

### Web 前端配置

编辑 `Web_Frontend/config.py`：

```python
# 推理 API 地址
API_BASE_URL = "http://localhost:9000"

# AI 分析配置（可选）
AI_API_KEY = "your-api-key"
AI_API_BASE = "https://api.longcat.chat/anthropic"
AI_MODEL = "LongCat-Flash-Lite"
AI_FORMAT = "anthropic"

# 数据库配置
SQLALCHEMY_DATABASE_URI = 'sqlite:///wad.db'
```

### 推理服务配置

编辑 `Inference_API/api_server.py`：

```python
# 修改监听地址和端口
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9000)
```

---

## 下一步

- 查看 [README.md](README.md) 了解完整功能
- 查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何贡献代码
- 查看 [CHANGELOG.md](CHANGELOG.md) 了解版本更新历史
- 提交 [Issue](../../issues) 报告问题或建议

---

## 获取帮助

如果遇到问题：

1. 查看本文档的"常见问题"部分
2. 搜索 [Issues](../../issues) 看是否有类似问题
3. 创建新的 [Issue](../../issues) 描述你的问题

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。