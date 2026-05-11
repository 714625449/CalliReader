#!/usr/bin/env python3
"""合并多个草书数据集的训练集到一个目录，用符号链接避免复制"""
import os
import sys
from pathlib import Path

# 输出目录
MERGED_DIR = Path('/root/sj-tmp/datasets/CaoshuMerged/Training')
MERGED_DIR.mkdir(parents=True, exist_ok=True)

# 源数据集（按优先级排序）
SOURCES = [
    '/root/sj-tmp/datasets/CCC_split/Training',
    '/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset/Training',
    '/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset_v2/Training',
]

total_links = 0
total_chars = set()

for src in SOURCES:
    src_path = Path(src)
    if not src_path.exists():
        print(f"Skip missing: {src}")
        continue
    print(f"\nProcessing: {src}")
    for char_dir in sorted(src_path.iterdir()):
        if not char_dir.is_dir():
            continue
        char_name = char_dir.name
        total_chars.add(char_name.rstrip('0123456789'))
        
        # 创建目标字符目录
        dst_char_dir = MERGED_DIR / char_name
        dst_char_dir.mkdir(exist_ok=True)
        
        for img_path in sorted(char_dir.glob('*.jpg')):
            dst_link = dst_char_dir / img_path.name
            if not dst_link.exists():
                os.symlink(img_path, dst_link)
                total_links += 1

print(f"\n{'='*50}")
print(f"Merged dataset: {MERGED_DIR}")
print(f"Total symlinks: {total_links}")
print(f"Unique chars: {len(total_chars)}")
print(f"Done.")
