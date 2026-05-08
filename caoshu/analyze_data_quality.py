"""
CaoshuReader - 数据质量分析
检查训练集和验证集的分布差异、噪声、标注问题
"""

import os
import sys
from pathlib import Path
import torch
from collections import Counter, defaultdict
from PIL import Image
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = '/workspace/CalliReader'
sys.path.insert(0, PROJECT_ROOT)

from caoshu.dataset import CaoshuDataset, get_transform


def analyze_overlap_and_distribution():
    """分析训练集和验证集的重叠和分布"""
    print("=" * 60)
    print("数据分布分析")
    print("=" * 60)

    data_root = '/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset'

    # 加载数据集
    train_dataset = CaoshuDataset(data_root, split='Training', transform=get_transform('Training'))
    val_dataset = CaoshuDataset(data_root, split='Validation', transform=get_transform('Validation'))

    print(f"\n训练集: {len(train_dataset)} 样本, {train_dataset.num_classes} 类别")
    print(f"验证集: {len(val_dataset)} 样本, {val_dataset.num_classes} 类别")

    # 类别重叠分析
    train_chars = set(train_dataset.char2idx.keys())
    val_chars = set(val_dataset.char2idx.keys())
    overlap_chars = train_chars & val_chars
    train_only = train_chars - val_chars
    val_only = val_chars - train_chars

    print(f"\n类别重叠情况:")
    print(f"  训练集特有: {len(train_only)} 字")
    print(f"  交集: {len(overlap_chars)} 字")
    print(f"  验证集特有: {len(val_only)} 字")

    if len(train_only) > 0:
        print(f"  训练集特有前10个字: {list(train_only)[:10]}")
    if len(val_only) > 0:
        print(f"  验证集特有前10个字: {list(val_only)[:10]}")

    # 每类样本分布
    train_label_counts = Counter([label for _, label in train_dataset.samples])
    val_label_counts = Counter([label for _, label in val_dataset.samples])

    train_samples_per_class = list(train_label_counts.values())
    val_samples_per_class = list(val_label_counts.values())

    print(f"\n每类样本数统计:")
    print(f"  训练集 - 平均: {np.mean(train_samples_per_class):.1f}, "
          f"中位数: {np.median(train_samples_per_class):.1f}, "
          f"最小: {min(train_samples_per_class)}, 最大: {max(train_samples_per_class)}")
    print(f"  验证集 - 平均: {np.mean(val_samples_per_class):.1f}, "
          f"中位数: {np.median(val_samples_per_class):.1f}, "
          f"最小: {min(val_samples_per_class)}, 最大: {max(val_samples_per_class)}")

    # 长尾分布分析
    train_threshold_10 = sum(1 for c in train_samples_per_class if c < 10)
    train_threshold_50 = sum(1 for c in train_samples_per_class if c < 50)
    val_threshold_5 = sum(1 for c in val_samples_per_class if c < 5)

    print(f"\n长尾分布:")
    print(f"  训练集 <10样本: {train_threshold_10} classes ({100 * train_threshold_10 / len(train_samples_per_class):.1f}%)")
    print(f"  训练集 <50样本: {train_threshold_50} classes ({100 * train_threshold_50 / len(train_samples_per_class):.1f}%)")
    print(f"  验证集 <5样本: {val_threshold_5} classes ({100 * val_threshold_5 / len(val_samples_per_class):.1f}%)")

    return train_dataset, val_dataset, overlap_chars


def analyze_image_quality(dataset, split_name, sample_size=2000):
    """分析图像质量: 分辨率、对比度、空白边距等"""
    print("\n" + "=" * 60)
    print(f"{split_name} 图像质量分析 (随机{sample_size}样本)")
    print("=" * 60)

    sample_indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)

    widths, heights = [], []
    aspect_ratios = []
    mean_intensities = []
    std_intensities = []
    blank_margin_ratios = []

    for idx in tqdm(sample_indices, desc=f"分析{split_name}图像质量"):
        img_path, label = dataset.samples[idx]
        try:
            img = Image.open(img_path).convert('L')  # 转为灰度
            img_np = np.array(img)

            # 基本尺寸
            w, h = img.size
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)

            # 强度统计
            mean_intensities.append(img_np.mean())
            std_intensities.append(img_np.std())

            # 空白边距比例 (白色像素 > 250)
            margin_mask = img_np > 250
            blank_ratio = margin_mask.mean()
            blank_margin_ratios.append(blank_ratio)

        except Exception as e:
            print(f"警告: 无法读取 {img_path}: {e}")

    # 统计分析
    print(f"\n{split_name} 图像质量统计:")
    print(f"  分辨率 - 宽: {np.mean(widths):.0f}±{np.std(widths):.0f}, "
          f"高: {np.mean(heights):.0f}±{np.std(heights):.0f}")
    print(f"  宽高比: {np.mean(aspect_ratios):.2f}±{np.std(aspect_ratios):.2f}")
    print(f"  平均强度: {np.mean(mean_intensities):.1f}±{np.std(mean_intensities):.1f} (0-255)")
    print(f"  对比度(std): {np.mean(std_intensities):.1f}±{np.std(std_intensities):.1f}")
    print(f"  空白边距比例: {np.mean(blank_margin_ratios):.3f}±{np.std(blank_margin_ratios):.3f} (0-1)")

    # 问题样本识别
    low_contrast = sum(1 for std in std_intensities if std < 20)
    high_blank = sum(1 for ratio in blank_margin_ratios if ratio > 0.7)

    print(f"\n潜在问题:")
    print(f"  低对比度图像(<20): {low_contrast} ({100 * low_contrast / len(std_intensities):.1f}%)")
    print(f"  过大边距图像(>0.7): {high_blank} ({100 * high_blank / len(blank_margin_ratios):.1f}%)")


def detect_noisy_labels(dataset, split_name, top_n=10):
    """检测可能的标签噪声：多版本字符之间的一致性"""
    print("\n" + "=" * 60)
    print(f"{split_name} 标签噪声检测")
    print("=" * 60)

    # 按去掉数字后缀前的原始目录名统计
    char_dir_samples = defaultdict(list)
    for path, label in dataset.samples:
        char_dir = Path(path).parent.name  # 如 "哀1", "哀2"
        char_dir_samples[char_dir].append((path, label))

    # 分析每个字符的多个版本之间的一致性
    print(f"\n字符多版本分析:")
    print(f"  总原始目录数: {len(char_dir_samples)}")

    multi_version_chars = {}
    for char_dir, samples in char_dir_samples.items():
        base_char = char_dir.rstrip('0123456789')
        if base_char not in multi_version_chars:
            multi_version_chars[base_char] = []
        multi_version_chars[base_char].extend(samples)

    # 只关注有多个版本的字符
    multi_version_chars = {k: v for k, v in multi_version_chars.items() if len(set(Path(s[0]).parent.name for s in v)) > 1}
    print(f"  有多版本的字符数: {len(multi_version_chars)}")
    print(f"  平均每字符版本数: {np.mean([len(set(Path(s[0]).parent.name for s in v)) for v in multi_version_chars.values()]):.1f}")

    # 潜在问题：样本数过少的字符
    few_samples_chars = [(char, len(samples)) for char, samples in multi_version_chars.items() if len(samples) < 5]
    if few_samples_chars:
        few_samples_chars.sort(key=lambda x: x[1])
        print(f"\n样本数过少的字符(<5样本): {len(few_samples_chars)}个")
        print(f"  前10个最少样本字符: {few_samples_chars[:10]}")

    # 某些字符可能视觉上相似，但标签不同
    return few_samples_chars


def main():
    print("CaoshuReader 数据质量完整分析")
    print("=" * 60)

    # 1. 分布和重叠分析
    train_dataset, val_dataset, overlap_chars = analyze_overlap_and_distribution()

    # 2. 图像质量分析
    analyze_image_quality(train_dataset, "Training", sample_size=2000)
    analyze_image_quality(val_dataset, "Validation", sample_size=1000)

    # 3. 噪声标签检测
    train_noisy = detect_noisy_labels(train_dataset, "Training")
    val_noisy = detect_noisy_labels(val_dataset, "Validation")

    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
