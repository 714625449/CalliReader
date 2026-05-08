"""
CaoshuReader - 修复训练和验证集划分问题
原问题：训练集和验证集类别严重不匹配，验证集样本数过少
解决方案：重新划分数据集，确保训练和验证覆盖相同字符集
"""

import os
import sys
import shutil
from pathlib import Path
import random
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm


def strip_suffix(dirname: str) -> str:
    """哀1 → 哀，哀12 → 哀，阿 → 阿"""
    import re
    return re.sub(r'\d+$', '', dirname)


def analyze_current_split():
    """分析当前数据集划分问题"""
    data_root = Path('/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset')

    # 按真实字符统计
    char_to_samples = defaultdict(list)
    char_to_dirs = defaultdict(list)

    for split in ['Training', 'Validation']:
        split_dir = data_root / split
        for char_dir in split_dir.iterdir():
            if not char_dir.is_dir():
                continue
            base_char = strip_suffix(char_dir.name)
            img_paths = list(char_dir.glob('*.jpg'))
            char_to_dirs[base_char].append((split, char_dir.name))
            char_to_samples[(base_char, split)].extend(img_paths)

    # 找出问题
    chars_in_training = {char for char, split in char_to_samples.keys() if split == 'Training'}
    chars_in_validation = {char for char, split in char_to_samples.keys() if split == 'Validation'}

    overlap = chars_in_training & chars_in_validation
    train_only = chars_in_training - chars_in_validation
    val_only = chars_in_validation - chars_in_training

    print("=" * 60)
    print("当前数据集划分问题")
    print("=" * 60)
    print(f"训练集字符数: {len(chars_in_training)}")
    print(f"验证集字符数: {len(chars_in_validation)}")
    print(f"交集字符数: {len(overlap)}")
    print(f"训练集特有: {len(train_only)} 字")
    print(f"验证集特有: {len(val_only)} 字")
    print()

    # 验证集样本数统计
    val_sample_counts = [len(char_to_samples[(char, 'Validation')]) for char in overlap]
    print(f"验证集样本分布 (交集字符):")
    print(f"  平均: {np.mean(val_sample_counts):.1f}")
    print(f"  最小: {min(val_sample_counts)}")
    print(f"  最大: {max(val_sample_counts)}")
    print(f"  < 5样本的字符: {sum(1 for c in val_sample_counts if c < 5)} "
          f"({100 * sum(1 for c in val_sample_counts if c < 5) / len(val_sample_counts):.1f}%)")
    print(f"  == 1样本的字符: {sum(1 for c in val_sample_counts if c == 1)} "
          f"({100 * sum(1 for c in val_sample_counts if c == 1) / len(val_sample_counts):.1f}%)")

    return char_to_dirs, char_to_samples


def create_balanced_split(min_val_samples=10, val_ratio=0.1):
    """
    创建新的平衡划分
    min_val_samples: 每类在验证集最少样本数
    val_ratio: 从总样本中划分给验证集的比例
    """
    data_root = Path('/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset')

    # 收集所有样本
    all_samples = defaultdict(list)  # char -> list of (original_path, version_dir)

    for split in ['Training', 'Validation']:
        split_dir = data_root / split
        for char_dir in split_dir.iterdir():
            if not char_dir.is_dir():
                continue
            base_char = strip_suffix(char_dir.name)
            version_dir = char_dir.name
            for img_path in char_dir.glob('*.jpg'):
                all_samples[base_char].append((img_path, version_dir))

    total_chars = len(all_samples)
    print(f"\n总字符数: {total_chars}")

    # 每个字符的最小验证样本数
    train_samples, val_samples = [], []

    valid_chars = 0  # 有足够样本的字符
    total_all_samples = 0

    for char, samples in all_samples.items():
        total_all_samples += len(samples)
        n_total = len(samples)

        # 如果总样本数足够，划分验证集
        if n_total >= min_val_samples * 2:  # 至少要有min_val_samples * 2才能划分
            n_val = max(min_val_samples, int(n_total * val_ratio))
            n_val = min(n_val, n_total - min_val_samples)  # 确保训练集也至少有min_val_samples

            # 随机选择验证样本
            random.shuffle(samples)
            val_samples.extend(samples[:n_val])
            train_samples.extend(samples[n_val:])
            valid_chars += 1
        else:
            # 样本太少，全部放入训练集
            train_samples.extend(samples)

    print(f"有足够样本的字符: {valid_chars} ({100 * valid_chars / total_chars:.1f}%)")
    print(f"总样本数: {total_all_samples}")
    print(f"训练集样本: {len(train_samples)} ({100 * len(train_samples) / total_all_samples:.1f}%)")
    print(f"验证集样本: {len(val_samples)} ({100 * len(val_samples) / total_all_samples:.1f}%)")

    # 验证：检查重叠
    train_chars = set(strip_suffix(Path(p).parent.name) for p, _ in train_samples)
    val_chars = set(strip_suffix(Path(p).parent.name) for p, _ in val_samples)
    overlap = train_chars & val_chars

    print(f"\n新划分后:")
    print(f"  训练集字符: {len(train_chars)}")
    print(f"  验证集字符: {len(val_chars)}")
    print(f"  交集: {len(overlap)}")
    print(f"  训练集特有: {len(train_chars - val_chars)}")
    print(f"  验证集特有: {len(val_chars - train_chars)}")

    # 准备新目录
    new_data_root = Path('/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset_v2')
    new_train_dir = new_data_root / 'Training'
    new_val_dir = new_data_root / 'Validation'

    if new_data_root.exists():
        print(f"\n警告: {new_data_root} 已存在，先删除...")
        import shutil
        shutil.rmtree(new_data_root)

    new_train_dir.mkdir(parents=True, exist_ok=True)
    new_val_dir.mkdir(parents=True, exist_ok=True)

    # 复制文件到新位置
    print("\n开始复制文件...")

    # 训练集
    print(f"复制训练集 ({len(train_samples)} 文件)...")
    for img_path, version_dir in tqdm(train_samples, desc="Training"):
        dest_dir = new_train_dir / version_dir
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)

    # 验证集
    print(f"复制验证集 ({len(val_samples)} 文件)...")
    for img_path, version_dir in tqdm(val_samples, desc="Validation"):
        dest_dir = new_val_dir / version_dir
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)

    print(f"\n新数据集创建完成: {new_data_root}")

    return new_data_root


def create_symlink_to_new_dataset():
    """创建符号链接，让train.py无感知使用新数据集"""
    original_path = '/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset'
    new_path = '/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset_v2'
    backup_path = '/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset_backup'

    original = Path(original_path)
    new = Path(new_path)

    if not new.exists():
        print(f"错误: 新数据集不存在: {new}")
        return False

    # 备份原数据
    if original.exists():
        print(f"备份原数据集到 {backup_path}...")
        original.rename(backup_path)

    # 创建符号链接
    print(f"创建符号链接: {original} -> {new}")
    original.symlink_to(new)

    print("数据集切换完成，train.py现在会使用新数据集")
    return True


def main():
    print("=" * 60)
    print("CaoshuReader 数据集修复工具")
    print("=" * 60)

    # 1. 分析当前问题
    analyze_current_split()

    # 2. 创建新划分
    print("\n" + "=" * 60)
    print("创建新的数据集划分...")
    print("=" * 60)
    new_data_root = create_balanced_split(min_val_samples=10, val_ratio=0.15)

    # 3. 切换数据
    print("\n" + "=" * 60)
    print("切换到新数据集...")
    print("=" * 60)
    create_symlink_to_new_dataset()

    print("\n" + "=" * 60)
    print("完成！请重新运行 train.py 使用修复后的数据集")
    print("=" * 60)


if __name__ == '__main__':
    main()
