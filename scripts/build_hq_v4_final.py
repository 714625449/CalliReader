#!/usr/bin/env python3
"""
Build CaoshuHQ_v4 final dataset:
1. SFZD: 443 train authors -> Training/ (528->448 direct resize)
2. SFZD: 50 val authors -> Validation/ (528->448 direct resize)
3. CCC raw originals: extract chars NOT in Training, resize 96->448
"""

import os
import csv
import shutil
import re
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
from tqdm import tqdm

SFZD_ROOT = Path('/root/sj-tmp/datasets/SFZD')
CCC_ROOT = Path('/root/sj-tmp/datasets/CCC/CCC_split')
HQ_ROOT = Path('/root/sj-tmp/datasets/CaoshuHQ_v4')

def resize_and_save(src: Path, dst: Path, size=(448, 448)):
    """Resize image to target size and save as RGB JPEG."""
    try:
        img = Image.open(src)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(size, Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"Error processing {src}: {e}")
        return False

# ========== Step 0: Load author splits ==========
print("[Step 0] Loading author splits...")
train_authors = set((SFZD_ROOT / 'train_authors.txt').read_text().strip().split('\n'))
val_authors = set((SFZD_ROOT / 'val_authors.txt').read_text().strip().split('\n'))
print(f"Train authors: {len(train_authors)}, Val authors: {len(val_authors)}")

# ========== Step 1: Build Training & Validation from SFZD ==========
print("[Step 1] Building Training & Validation from SFZD...")

# Read CSV to get exact author for each image
train_samples = []  # list of (char, src_path, dst_name)
val_samples = []

with open(SFZD_ROOT / 'labels_with_size.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        author = row['作者']
        char = row['字']
        pic_id = row['图片标识']  # e.g., "K/0"
        if '/' not in pic_id:
            continue
        code, img_name = pic_id.split('/')
        src = SFZD_ROOT / 'c' / code / f"{img_name}.png"
        
        if not src.exists():
            continue
        
        # Unique dst name: {code}_{img_name}.jpg
        dst_name = f"{code}_{img_name}.jpg"
        
        if author in train_authors:
            train_samples.append((char, src, dst_name))
        elif author in val_authors:
            val_samples.append((char, src, dst_name))

print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

# Copy & resize train
print("[Step 1a] Processing training set...")
train_root = HQ_ROOT / 'Training'
train_count = 0
for char, src, dst_name in tqdm(train_samples, desc="Training"):
    dst = train_root / char / dst_name
    if resize_and_save(src, dst):
        train_count += 1

print(f"Train images saved: {train_count}")

# Copy & resize val
print("[Step 1b] Processing validation set...")
val_root = HQ_ROOT / 'Validation'
val_count = 0
for char, src, dst_name in tqdm(val_samples, desc="Validation"):
    dst = val_root / char / dst_name
    if resize_and_save(src, dst):
        val_count += 1

print(f"Val images saved: {val_count}")

# ========== Step 2: Extract CCC originals for chars not in Training ==========
print("[Step 2] Extracting CCC original images for uncovered chars...")

train_chars = set(d.name for d in train_root.iterdir() if d.is_dir())
print(f"Training chars: {len(train_chars)}")

training_dir = CCC_ROOT / 'Training'
ccc_raw_root = HQ_ROOT / 'Training_CCC'
ccc_samples = []

char_dirs = [d for d in training_dir.iterdir() if d.is_dir()]
print(f"CCC training char dirs: {len(char_dirs)}")

# Group by base char (strip trailing digits)
base_to_dirs = defaultdict(list)
for char_dir in char_dirs:
    base = re.sub(r'\d+$', '', char_dir.name)
    base_to_dirs[base].append(char_dir)

print(f"CCC base chars: {len(base_to_dirs)}")

for base, dirs in base_to_dirs.items():
    if base in train_chars:
        continue  # Skip chars already in training
    
    target_dir = ccc_raw_root / base
    
    for char_dir in dirs:
        for img_path in sorted(char_dir.glob('*.jpg')):
            # Skip generated/synthetic images
            if img_path.name.startswith('gen_'):
                continue
            
            # CCC original: resize 96x96 -> 448x448
            dst_name = f"{img_path.stem}_ccc.jpg"
            dst = target_dir / dst_name
            
            if resize_and_save(img_path, dst):
                ccc_samples.append((base, img_path.name))

print(f"CCC original samples for uncovered chars: {len(ccc_samples)}")

# ========== Step 3: Statistics ==========
print("\n" + "="*60)
print("CaoshuHQ_v4 Build Summary")
print("="*60)

train_stats = Counter()
for char_dir in sorted(train_root.iterdir()):
    if char_dir.is_dir():
        train_stats[char_dir.name] = len(list(char_dir.glob('*.jpg')))

val_stats = Counter()
for char_dir in sorted(val_root.iterdir()):
    if char_dir.is_dir():
        val_stats[char_dir.name] = len(list(char_dir.glob('*.jpg')))

ccc_stats = Counter()
for char_dir in sorted(ccc_raw_root.iterdir()):
    if char_dir.is_dir():
        ccc_stats[char_dir.name] = len(list(char_dir.glob('*.jpg')))

print(f"\nTraining (SFZD):")
print(f"  Chars: {len(train_stats)}, Images: {sum(train_stats.values())}")

print(f"\nValidation (SFZD):")
print(f"  Chars: {len(val_stats)}, Images: {sum(val_stats.values())}")

print(f"\nTraining_CCC (CCC originals, not in SFZD):")
print(f"  Chars: {len(ccc_stats)}, Images: {sum(ccc_stats.values())}")

# Top/bottom CCC chars
if ccc_stats:
    print(f"\nTop 10 CCC chars by count:")
    for char, count in ccc_stats.most_common(10):
        print(f"  {char}: {count}")
    print(f"\nBottom 10 CCC chars by count:")
    for char, count in ccc_stats.most_common()[-10:]:
        print(f"  {char}: {count}")

# Save summary
summary_path = HQ_ROOT / 'build_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("# CaoshuHQ_v4 Build Summary\n\n")
    f.write(f"## Training (SFZD 443 authors, 528->448)\n")
    f.write(f"- Chars: {len(train_stats)}\n")
    f.write(f"- Images: {sum(train_stats.values())}\n\n")
    f.write(f"## Validation (SFZD 50 authors, 528->448)\n")
    f.write(f"- Chars: {len(val_stats)}\n")
    f.write(f"- Images: {sum(val_stats.values())}\n\n")
    f.write(f"## Training_CCC (CCC originals 96->448, chars not in SFZD)\n")
    f.write(f"- Chars: {len(ccc_stats)}\n")
    f.write(f"- Images: {sum(ccc_stats.values())}\n\n")
    if ccc_stats:
        f.write("Char list:\n")
        for char, count in sorted(ccc_stats.items()):
            f.write(f"  {char}: {count}\n")

print(f"\n[Done] Summary saved to {summary_path}")

# Disk usage
du_result = os.popen(f"du -sh {HQ_ROOT}").read().strip()
print(f"Total disk usage: {du_result}")
