"""
数据集划分脚本
将 payload_train.csv 划分为训练集和验证集
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

# ============================================================
# 可配置参数
# ============================================================
INPUT_FILE = "payload_train.csv"          # 输入文件路径
TRAIN_OUTPUT = "train_split.csv"          # 训练集输出路径
VAL_OUTPUT = "val_split.csv"              # 验证集输出路径
LOG_OUTPUT = "split_log.txt"              # 日志输出路径
TEST_SIZE = 0.2                           # 验证集比例
RANDOM_SEED = 42                          # 随机种子
REQUIRED_COLUMNS = ["payload", "attack_type"]  # 必需列名
VALID_CLASSES = ["norm", "sqli", "xss", "path-traversal", "cmdi"]  # 合法类别
# ============================================================


def setup_seed(seed):
    """设置全局随机种子，确保可复现"""
    np.random.seed(seed)


def validate_input(df, required_columns):
    """校验输入数据合法性"""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"输入数据缺少必需列: {missing_cols}")


def validate_classes(df, column, valid_classes):
    """校验类别是否合法"""
    invalid_classes = set(df[column].unique()) - set(valid_classes)
    if invalid_classes:
        raise ValueError(f"发现非法类别: {invalid_classes}")


def validate_val_set(train_classes, val_classes):
    """强制检查验证集是否包含所有类别"""
    missing_classes = train_classes - val_classes
    if missing_classes:
        raise ValueError(f"验证集缺少类别: {missing_classes}，划分失败!")


def split_dataset():
    """执行数据集划分"""
    # 设置随机种子
    setup_seed(RANDOM_SEED)

    # 读取数据
    print(f"正在读取 {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")

    # 输入数据校验
    validate_input(df, REQUIRED_COLUMNS)
    validate_classes(df, "attack_type", VALID_CLASSES)

    # 统计原始数据
    class_counts = df["attack_type"].value_counts()
    total_samples = len(df)

    # 分层抽样划分
    train_data, val_data = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["attack_type"]
    )

    # 验证集完整性强制检查
    train_classes = set(train_data["attack_type"].unique())
    val_classes = set(val_data["attack_type"].unique())
    validate_val_set(train_classes, val_classes)

    # 保存文件
    train_data.to_csv(TRAIN_OUTPUT, index=False, encoding="utf-8")
    val_data.to_csv(VAL_OUTPUT, index=False, encoding="utf-8")

    # 生成日志
    log_content = generate_log(df, train_data, val_data, class_counts, total_samples, train_classes, val_classes)

    # 保存日志
    with open(LOG_OUTPUT, "w", encoding="utf-8") as f:
        f.write(log_content)

    # 打印结果
    print(log_content)
    print(f"\n日志已保存至: {LOG_OUTPUT}")


def generate_log(df, train_data, val_data, class_counts, total_samples, train_classes, val_classes):
    """生成详细日志"""
    train_counts = train_data["attack_type"].value_counts()
    val_counts = val_data["attack_type"].value_counts()

    log_lines = [
        "=" * 60,
        "数据集划分日志",
        "=" * 60,
        f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "【配置参数】",
        f"  输入文件: {INPUT_FILE}",
        f"  训练集输出: {TRAIN_OUTPUT}",
        f"  验证集输出: {VAL_OUTPUT}",
        f"  验证集比例: {TEST_SIZE}",
        f"  随机种子: {RANDOM_SEED}",
        "",
        "【原始数据统计】",
        f"  总样本数: {total_samples}",
    ]

    for cls, count in class_counts.items():
        log_lines.append(f"  {cls}: {count} ({count/total_samples*100:.1f}%)")

    log_lines.extend([
        "",
        "【划分结果】",
        f"  训练集: {len(train_data)} 条",
        f"  验证集: {len(val_data)} 条",
        "",
        "【训练集类别分布】",
    ])

    for cls, count in train_counts.items():
        log_lines.append(f"  {cls}: {count} ({count/len(train_data)*100:.1f}%)")

    log_lines.append("\n【验证集类别分布】")
    for cls, count in val_counts.items():
        log_lines.append(f"  {cls}: {count} ({count/len(val_data)*100:.1f}%)")

    log_lines.extend([
        "",
        "【完整性检查】",
        f"  训练集类别数: {len(train_classes)}",
        f"  验证集类别数: {len(val_classes)}",
        f"  缺失类别: 无",
        "",
        "=" * 60,
    ])

    return "\n".join(log_lines)


if __name__ == "__main__":
    split_dataset()
