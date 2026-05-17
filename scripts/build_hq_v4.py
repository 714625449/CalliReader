#!/usr/bin/env python3
"""Build HQ dataset from SFZD + CCC raw images"""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict, Counter

SFZD_ROOT = Path('/root/sj-tmp/datasets/SFZD')
CCC_ROOT = Path('/root/sj-tmp/datasets/CCC/CCC_split')
HQ_ROOT = Path('/root/sj-tmp/datasets/HQ')
CCC_RAW = Path('/root/sj-tmp/datasets/CCC/CCC_split/raw')
CCC_ONLY = Path('/root/sj-tmp/datasets/CCC/CCC_split/raw/CCC_only')

# ========== Step 1: Parse SQL and build HQ from SFZD ==========
print("[Step 1] Parsing SFZD SQL...")

sql_path = SFZD_ROOT / 'czb_sf_c.sql'
with open(sql_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Extract all tuples
pattern = r"\((\d+),'([^']*)','([^']*)','([^']*)'\)"
matches = re.findall(pattern, content)

print(f"[Step 1] Total SQL records: {len(matches)}")

# Build hanzi -> list of (tupian, id)
hanzi_to_images = defaultdict(list)
for id_, hanzi, shufajia, tupian in matches:
    if hanzi and tupian:
        hanzi_to_images[hanzi].append((tupian, int(id_)))

print(f"[Step 1] Unique hanzi in SFZD: {len(hanzi_to_images)}")

# Copy images to HQ
HQ_ROOT.mkdir(parents=True, exist_ok=True)
copied = 0
missing = 0

for hanzi, images in sorted(hanzi_to_images.items()):
    char_dir = HQ_ROOT / hanzi
    char_dir.mkdir(exist_ok=True)
    
    for tupian, id_ in images:
        # tupian format: "N/1DM" -> author_code=N, img_id=1DM
        if '/' not in tupian:
            continue
        author_code, img_id = tupian.split('/')
        src = SFZD_ROOT / 'c' / author_code / f"{img_id}.png"
        
        if not src.exists():
            missing += 1
            continue
        
        # Ensure unique filename in char_dir
        dst_name = f"{img_id}.png"
        dst = char_dir / dst_name
        if dst.exists():
            dst_name = f"{img_id}_{id_}.png"
            dst = char_dir / dst_name
        
        shutil.copy2(src, dst)
        copied += 1

print(f"[Step 1] Copied: {copied}, Missing: {missing}")

# ========== Step 2: Process CCC Training raw images ==========
print("[Step 2] Processing CCC Training raw images...")

CCC_RAW.mkdir(parents=True, exist_ok=True)
CCC_ONLY.mkdir(parents=True, exist_ok=True)

training_dir = CCC_ROOT / 'Training'
if not training_dir.exists():
    print(f"[Step 2] ERROR: {training_dir} not found!")
else:
    # First, merge digit-suffix folders
    # e.g., "七", "七2", "七3" -> all merge to "七"
    char_dirs = [d for d in training_dir.iterdir() if d.is_dir()]
    
    # Group by base char (strip trailing digits)
    base_to_dirs = defaultdict(list)
    for char_dir in char_dirs:
        base = re.sub(r'\d+$', '', char_dir.name)
        base_to_dirs[base].append(char_dir)
    
    print(f"[Step 2] Merging {len(char_dirs)} dirs into {len(base_to_dirs)} base chars...")
    
    ccc_raw_counts = Counter()
    
    for base, dirs in sorted(base_to_dirs.items()):
        target_dir = CCC_RAW / base
        target_dir.mkdir(exist_ok=True)
        
        for char_dir in dirs:
            # Find original images (no "gen" in filename)
            for img_path in sorted(char_dir.glob('*.jpg')):
                if '_gen' in img_path.name:
                    continue
                
                # Rename: original_name_ccc.jpg
                new_name = f"{img_path.stem}_ccc.jpg"
                dst = target_dir / new_name
                
                # Handle collision
                counter = 1
                while dst.exists():
                    new_name = f"{img_path.stem}_{counter}_ccc.jpg"
                    dst = target_dir / new_name
                    counter += 1
                
                shutil.copy2(img_path, dst)
                ccc_raw_counts[base] += 1
    
    print(f"[Step 2] CCC raw images copied: {sum(ccc_raw_counts.values())}")

# ========== Step 3: Compare HQ vs CCC raw ==========
print("[Step 3] Comparing HQ vs CCC raw...")

hq_chars = set(d.name for d in HQ_ROOT.iterdir() if d.is_dir())
ccc_chars = set(d.name for d in CCC_RAW.iterdir() if d.is_dir() and d.name != 'CCC_only')

ccc_only_chars = ccc_chars - hq_chars
print(f"[Step 3] HQ chars: {len(hq_chars)}, CCC chars: {len(ccc_chars)}, CCC_only: {len(ccc_only_chars)}")

# Move CCC_only chars to CCC_ONLY directory
ccc_only_counts = Counter()
for char in sorted(ccc_only_chars):
    src = CCC_RAW / char
    dst = CCC_ONLY / char
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    ccc_only_counts[char] = len(list(dst.glob('*.jpg')))

print(f"[Step 3] CCC_only chars moved: {len(ccc_only_counts)}")

# ========== Step 4: Output statistics ==========
print("\n" + "="*60)
print("HQ Statistics")
print("="*60)

hq_counts = Counter()
for char_dir in sorted(HQ_ROOT.iterdir()):
    if not char_dir.is_dir():
        continue
    count = len(list(char_dir.glob('*.png')))
    hq_counts[char_dir.name] = count

print(f"Total chars: {len(hq_counts)}")
print(f"Total images: {sum(hq_counts.values())}")
print(f"\nTop 20 chars by count:")
for char, count in hq_counts.most_common(20):
    print(f"  {char}: {count}")
print(f"\nBottom 10 chars by count:")
for char, count in hq_counts.most_common()[-10:]:
    print(f"  {char}: {count}")

print("\n" + "="*60)
print("CCC_only Statistics")
print("="*60)

print(f"Total chars: {len(ccc_only_counts)}")
print(f"Total images: {sum(ccc_only_counts.values())}")
print(f"\nTop 20 CCC_only chars by count:")
for char, count in ccc_only_counts.most_common(20):
    print(f"  {char}: {count}")
print(f"\nBottom 10 CCC_only chars by count:")
for char, count in ccc_only_counts.most_common()[-10:]:
    print(f"  {char}: {count}")

# Save summary
with open('/root/sj-tmp/datasets/hq_build_summary.txt', 'w', encoding='utf-8') as f:
    f.write("# HQ Build Summary\n\n")
    f.write(f"## HQ (from SFZD)\n")
    f.write(f"- Total chars: {len(hq_counts)}\n")
    f.write(f"- Total images: {sum(hq_counts.values())}\n")
    f.write(f"- Char list:\n")
    for char, count in sorted(hq_counts.items()):
        f.write(f"  {char}: {count}\n")
    
    f.write(f"\n## CCC_only (chars in CCC but not in HQ)\n")
    f.write(f"- Total chars: {len(ccc_only_counts)}\n")
    f.write(f"- Total images: {sum(ccc_only_counts.values())}\n")
    f.write(f"- Char list:\n")
    for char, count in sorted(ccc_only_counts.items()):
        f.write(f"  {char}: {count}\n")

print(f"\n[Done] Summary saved to /root/sj-tmp/datasets/hq_build_summary.txt")
