# 贡献指南

感谢你对 WAD (Web Attack Detection System) 项目的关注！我们欢迎任何形式的贡献。

---

## 如何贡献

### 1. 报告 Bug

如果你发现了 bug，请通过以下方式报告：

1. 在 GitHub 上创建 [Issue](../../issues)
2. 使用 Bug 报告模板
3. 提供详细的复现步骤
4. 包含错误日志和截图（如果适用）

### 2. 建议新功能

我们欢迎新功能建议：

1. 在 GitHub 上创建 [Issue](../../issues)
2. 使用功能请求模板
3. 详细描述功能需求和使用场景

### 3. 提交代码

#### 准备工作

1. Fork 本仓库
2. 克隆你的 Fork：
   ```bash
   git clone https://github.com/your-username/Web-Attack-Detection-System-HttpParamsDataset.git
   ```
3. 创建新分支：
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### 开发规范

1. **代码风格**
   - Python 代码遵循 PEP 8 规范
   - 使用有意义的变量和函数名
   - 添加必要的注释

2. **提交信息**
   - 使用清晰的提交信息
   - 格式：`<类型>: <描述>`
   - 类型：`feat`（新功能）、`fix`（修复）、`docs`（文档）、`style`（格式）、`refactor`（重构）

3. **测试**
   - 确保现有测试通过
   - 为新功能添加测试
   - 测试覆盖率不低于 80%

4. **文档**
   - 更新 README.md（如果需要）
   - 添加代码注释
   - 更新 API 文档（如果适用）

#### 提交 Pull Request

1. 确保代码通过所有测试
2. 更新相关文档
3. 提交 Pull Request 到 `main` 分支
4. 在 PR 描述中说明修改内容

---

## 开发环境设置

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/Web-Attack-Detection-System-HttpParamsDataset.git
cd Web-Attack-Detection-System-HttpParamsDataset
```

### 2. 创建虚拟环境

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 使用 conda
conda create -n wad python=3.9
conda activate wad
```

### 3. 安装依赖

```bash
# 训练模块
cd Train_Model
pip install -r requirements.txt

# Web前端
cd ../Web_Frontend
pip install -r requirements.txt

# 推理服务
cd ../Inference_API
pip install -r requirements.txt
```

### 4. 启动服务

```bash
# 启动推理服务（终端1）
cd Inference_API
python api_server.py

# 启动Web前端（终端2）
cd ../Web_Frontend
python app.py
```

---

## 项目结构

```
Web-Attack-Detection-System-HttpParamsDataset/
├── Train_Model/                 # 模型训练部分
│   ├── train.py                 # 训练脚本
│   └── requirements.txt         # 依赖
│
├── Web_Frontend/                # Web前端部分
│   ├── app.py                   # Flask应用
│   └── requirements.txt         # 依赖
│
├── Inference_API/               # 后端推理部分
│   ├── api_server.py            # FastAPI服务
│   └── requirements.txt         # 依赖
│
├── CONTRIBUTING.md              # 本文件
├── LICENSE                      # 许可证
└── README.md                    # 项目说明
```

---

## 代码审查标准

### 1. 功能性

- 代码是否实现了预期功能？
- 是否有边界情况处理？
- 是否有错误处理？

### 2. 代码质量

- 代码是否易于理解？
- 是否有重复代码？
- 是否有适当的注释？

### 3. 性能

- 是否有性能问题？
- 是否有优化空间？
- 是否有内存泄漏？

### 4. 安全性

- 是否有安全漏洞？
- 是否有输入验证？
- 是否有敏感信息泄露？

---

## 常见问题

### Q: 如何添加新的攻击类型？

A: 参考 `train.py` 中的数据加载部分，添加新的标签和训练数据。

### Q: 如何优化模型性能？

A: 可以尝试：
- 调整超参数
- 增加训练数据
- 使用更复杂的模型架构
- 进行特征工程

### Q: 如何添加新的 API 接口？

A: 参考 `Web_Frontend/blueprints/api.py` 中的现有接口实现。

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](../../issues)
- 发送邮件至：1141606412@qq.com

---

## 致谢

感谢所有为本项目做出贡献的人！

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。