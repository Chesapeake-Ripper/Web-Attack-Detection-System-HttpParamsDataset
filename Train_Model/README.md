# WAD 模型训练模块

> Web Attack Detection System - 模型训练与评估  
> 包含 LightGBM 和 TextCNN 两种模型的训练代码

---

## 功能特性

- **数据预处理** — 自动加载和预处理 HttpParamsDataset 数据集
- **特征工程** — TF-IDF 特征提取（字符级 + 词级）
- **模型训练** — 支持 LightGBM 和 TextCNN 两种模型
- **模型评估** — 完整的评估指标和可视化
- **结果导出** — 保存训练好的模型和评估结果

---

## 技术栈

- **机器学习**: LightGBM, scikit-learn
- **深度学习**: PyTorch (TextCNN)
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn
- **特征提取**: TF-IDF (字符级 + 词级)

---

## 项目结构

```
Train_Model/
├── train.py                    # 主训练脚本
├── generate_paper_figures.py   # 论文图表生成脚本
├── requirements.txt            # 依赖列表
├── HttpParamsDataset/          # 数据集目录
│   ├── payload_train.csv       # 训练集
│   ├── payload_test.csv        # 测试集
│   ├── payload_full.csv        # 完整数据集
│   └── README.md               # 数据集说明
├── outputs/                    # 训练输出目录
│   ├── lgbm_model.txt          # LightGBM 模型
│   ├── textcnn_best.pt         # TextCNN 模型
│   ├── char_tfidf.pkl          # 字符级 TF-IDF 向量化器
│   ├── word_tfidf.pkl          # 词级 TF-IDF 向量化器
│   ├── label_encoder.pkl       # 标签编码器
│   ├── *.png                   # 各种图表
│   └── *.csv                   # 评估结果
├── 训练结果.md                  # 训练结果记录
├── 项目总结.md                  # 项目总结
└── README.md                   # 本文件
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
pip install torch scikit-learn lightgbm pandas numpy matplotlib seaborn joblib
```

### 开始训练

```bash
# 运行完整训练流程
python train.py

# 生成论文图表
python generate_paper_figures.py
```

---

## 训练流程

### 1. 数据加载

```python
# 自动加载 HttpParamsDataset 数据集
# 训练集: 20,712 条
# 测试集: 10,355 条
```

### 2. 特征工程

```python
# 字符级 TF-IDF 特征
char_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))

# 词级 TF-IDF 特征
word_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
```

### 3. 模型训练

#### LightGBM

```python
# 使用 TF-IDF 特征训练 LightGBM
lgbm = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31
)
```

#### TextCNN

```python
# 字符级卷积神经网络
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        # 多个不同大小的卷积核
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 100, kernel_size=k)
            for k in [3, 4, 5]
        ])
```

### 4. 模型评估

```python
# 评估指标
- Accuracy
- Precision
- Recall
- F1-Score (Macro/Weighted)
- 混淆矩阵
```

---

## 模型性能

测试集：HttpParamsDataset，共 **10,355** 条，5 分类

| 模型 | Accuracy | Precision | Recall | Macro F1 | Weighted F1 |
|------|----------|-----------|--------|----------|-------------|
| LightGBM | 99.95% | 98.18% | 99.67% | 98.88% | 99.95% |
| TextCNN | 99.94% | 99.27% | 97.99% | 98.61% | 99.94% |

### 各类别 F1-Score

| 类别 | 测试样本数 | LightGBM F1 | TextCNN F1 |
|------|-----------|-------------|-----------|
| 正常流量 | 6434 | 1.00 | 1.00 |
| SQL注入 | 3617 | 1.00 | 1.00 |
| XSS攻击 | 177 | 1.00 | 1.00 |
| 路径穿越 | 97 | 0.99 | 1.00 |
| 命令注入 | 30 | 0.95 | 0.93 |

---

## 输出文件说明

### 模型文件

- `lgbm_model.txt` — LightGBM 模型文件
- `textcnn_best.pt` — TextCNN 模型权重
- `char_tfidf.pkl` — 字符级 TF-IDF 向量化器
- `word_tfidf.pkl` — 词级 TF-IDF 向量化器
- `label_encoder.pkl` — 标签编码器

### 评估结果

- `model_comparison.csv` — 模型对比结果
- `model_comparison.png` — 模型对比图表
- `cm_lightgbm*.png` — LightGBM 混淆矩阵
- `cm_textcnn*.png` — TextCNN 混淆矩阵
- `lgbm_feature_importance.png` — 特征重要性
- `textcnn_training_curves.png` — 训练曲线

---

## 自定义训练

### 修改模型参数

编辑 `train.py` 中的模型参数：

```python
# LightGBM 参数
lgbm_params = {
    'n_estimators': 500,
    'learning_rate': 0.1,
    'max_depth': 6,
    'num_leaves': 31,
    'min_child_samples': 20
}

# TextCNN 参数
textcnn_params = {
    'embed_dim': 128,
    'num_filters': 100,
    'kernel_sizes': [3, 4, 5],
    'dropout': 0.5
}
```

### 使用自己的数据集

1. 准备 CSV 文件，包含 `payload` 和 `attack_type` 列
2. 修改 `train.py` 中的数据加载路径
3. 运行训练脚本

---

## 训练技巧

### 1. 处理类别不平衡

```python
# LightGBM 类别权重
class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
lgbm = lgb.LGBMClassifier(class_weight=dict(zip(classes, class_weights)))
```

### 2. 早停策略

```python
# TextCNN 早停
early_stopping = EarlyStopping(patience=10, min_delta=0.001)
```

### 3. 学习率调度

```python
# 学习率衰减
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
```

---

## 故障排除

### 内存不足

```bash
# 减少批处理大小
# 在 train.py 中修改 batch_size 参数
batch_size = 256  # 默认可能是 512 或更大
```

### 训练速度慢

```bash
# 1. 使用 GPU 加速 (需要 CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 减少数据量进行测试
# 使用部分数据集进行快速验证
```

### 模型加载失败

```bash
# 检查输出文件是否存在
ls -la outputs/

# 检查依赖版本
pip list | grep -E "torch|lightgbm|scikit-learn"
```

---

## 引用

如果使用了 HttpParamsDataset 数据集，请引用：

```bibtex
@dataset{http_params_dataset,
  title={HttpParamsDataset: HTTP Request Parameter Values Dataset},
  year={2024},
  publisher={GitHub},
  url={https://github.com/your-username/HttpParamsDataset}
}
```

---

## 许可证

MIT License