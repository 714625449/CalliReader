#!/usr/bin/env python3
"""Track 1: 按作者名划分 shufazidian 493 位作者为 443 训练 + 50 验证，并建验证集 A 目录"""

import csv
import random
import os
from pathlib import Path
from collections import defaultdict

random.seed(42)

# 读取 CSV，按作者分组
author_to_samples = defaultdict(list)
all_authors = set()

with open('/root/sj-tmp/datasets/shufazidian/labels_with_size.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        author = row['作者']
        char = row['字']
        pic_id = row['图片标识']
        author_code, img_name = pic_id.split('/')
        img_path = f"/root/sj-tmp/datasets/shufazidian/c/{author_code}/{img_name}.png"
        
        if os.path.exists(img_path):
            author_to_samples[author].append({
                'char': char,
                'img_path': img_path,
                'pic_id': pic_id
            })
            all_authors.add(author)

authors = sorted(list(all_authors))
print(f"[Track1] Total authors with valid images: {len(authors)}")

random.shuffle(authors)
train_authors = authors[:443]
val_authors = authors[443:]

print(f"[Track1] Train authors: {len(train_authors)}")
print(f"[Track1] Val authors: {len(val_authors)}")

# 保存作者列表
os.makedirs('/root/sj-tmp/datasets/shufazidian', exist_ok=True)
with open('/root/sj-tmp/datasets/shufazidian/train_authors.txt', 'w') as f:
    for a in train_authors:
        f.write(a + '\n')
with open('/root/sj-tmp/datasets/shufazidian/val_authors.txt', 'w') as f:
    for a in val_authors:
        f.write(a + '\n')

# 建主验证集 A 目录（按字分类，符号链接）
val_root = Path('/root/sj-tmp/datasets/Validation_Real/main_block')
val_root.mkdir(parents=True, exist_ok=True)

val_count = 0
val_chars = set()
for author in val_authors:
    for sample in author_to_samples[author]:
        char = sample['char']
        src = sample['img_path']
        char_dir = val_root / char
        char_dir.mkdir(exist_ok=True)
        
        # 符号链接（避免复制 5K 张图）
        dst = char_dir / f"{author}_{Path(src).name}"
        if not dst.exists():
            os.symlink(src, dst)
        val_count += 1
        val_chars.add(char)

print(f"[Track1] Val samples: {val_count}")
print(f"[Track1] Val chars: {len(val_chars)}")
print(f"[Track1] Val root: {val_root}")

# 统计训练集样本数
train_count = 0
for author in train_authors:
    train_count += len(author_to_samples[author])
print(f"[Track1] Train samples: {train_count}")
