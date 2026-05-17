#!/usr/bin/env python3
"""
构建最终数据集：
- HQ: 每字保底划分(train/val/test)，全部 resize 448x448 保存
- CCC_only: 每字随机抽 20 张，同样划分，文件名加 _ccc
"""
import os
import shutil
import random
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 路径
HQ_SOURCE = Path('/root/sj-tmp/datasets/HQ_source')
CCC_SOURCE = Path('/root/sj-tmp/datasets/CCC_only')
HQ_OUT = Path('/root/sj-tmp/datasets/HQ')

TARGET_SIZE = (448, 448)
random.seed(42)

def resize_and_save(src_path, dst_path):
    """读取图片，resize 到 448x448，保存为 JPEG"""
    try:
        with Image.open(src_path) as img:
            img = img.convert('RGB')
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            img.save(dst_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def split_char_files(char_dir):
    """对一个字的目录，按保底策略划分 train/val/test"""
    files = sorted([f for f in char_dir.iterdir() if f.is_file()])
    files = [f for f in files if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.gif')]
    random.shuffle(files)
    
    n = len(files)
    if n == 0:
        return [], [], []
    elif n == 1:
        return files, [], []
    elif n == 2:
        return files[:1], files[1:2], []
    else:
        # 保底: val 1, test 1, 剩余 train
        val_files = files[:1]
        test_files = files[1:2]
        train_files = files[2:]
        return train_files, val_files, test_files

def process_hq_char(args):
    """处理一个 HQ 字目录"""
    char_dir, char_name = args
    train_files, val_files, test_files = split_char_files(char_dir)
    
    results = {'char': char_name, 'train': 0, 'val': 0, 'test': 0}
    
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        if not files:
            continue
        out_dir = HQ_OUT / split / char_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = out_dir / f"{src.stem}.jpg"
            if resize_and_save(src, dst):
                results[split] += 1
    
    return results

def process_ccc_char(args):
    """处理一个 CCC_only 字目录：抽 20 张，划分，保存"""
    char_dir, char_name = args
    files = sorted([f for f in char_dir.iterdir() if f.is_file()])
    files = [f for f in files if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.gif')]
    
    # 随机抽 20 张
    if len(files) > 20:
        files = random.sample(files, 20)
    
    train_files, val_files, test_files = split_char_files(char_dir)
    # 重新用抽样的 files 划分
    random.shuffle(files)
    n = len(files)
    if n == 0:
        return {'char': char_name, 'train': 0, 'val': 0, 'test': 0, 'source': 'ccc'}
    elif n == 1:
        train_files, val_files, test_files = files, [], []
    elif n == 2:
        train_files, val_files, test_files = files[:1], files[1:2], []
    else:
        val_files = files[:1]
        test_files = files[1:2]
        train_files = files[2:]
    
    results = {'char': char_name, 'train': 0, 'val': 0, 'test': 0, 'source': 'ccc'}
    
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        if not files:
            continue
        out_dir = HQ_OUT / split / char_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = out_dir / f"{src.stem}_ccc.jpg"
            if resize_and_save(src, dst):
                results[split] += 1
    
    return results

def main():
    print("=" * 60)
    print("Building final dataset")
    print("=" * 60)
    
    # 清理旧的 train/val/test（如果存在）
    for split in ['train', 'val', 'test']:
        d = HQ_OUT / split
        if d.exists():
            for sub in d.iterdir():
                if sub.is_dir():
                    shutil.rmtree(sub)
    
    # 处理 HQ
    print("\nProcessing HQ...")
    hq_chars = sorted([d for d in HQ_SOURCE.iterdir() if d.is_dir()])
    print(f"HQ chars: {len(hq_chars)}")
    
    hq_results = []
    for char_dir in hq_chars:
        result = process_hq_char((char_dir, char_dir.name))
        hq_results.append(result)
    
    # 处理 CCC_only
    print("\nProcessing CCC_only...")
    ccc_chars = sorted([d for d in CCC_SOURCE.iterdir() if d.is_dir()])
    print(f"CCC_only chars: {len(ccc_chars)}")
    
    ccc_results = []
    for char_dir in ccc_chars:
        result = process_ccc_char((char_dir, char_dir.name))
        ccc_results.append(result)
    
    # 统计
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    
    hq_train = sum(r['train'] for r in hq_results)
    hq_val = sum(r['val'] for r in hq_results)
    hq_test = sum(r['test'] for r in hq_results)
    
    ccc_train = sum(r['train'] for r in ccc_results)
    ccc_val = sum(r['val'] for r in ccc_results)
    ccc_test = sum(r['test'] for r in ccc_results)
    
    print(f"\nHQ:")
    print(f"  Train: {hq_train} images")
    print(f"  Val:   {hq_val} images")
    print(f"  Test:  {hq_test} images")
    print(f"  Total: {hq_train + hq_val + hq_test} images")
    
    print(f"\nCCC_only:")
    print(f"  Train: {ccc_train} images")
    print(f"  Val:   {ccc_val} images")
    print(f"  Test:  {ccc_test} images")
    print(f"  Total: {ccc_train + ccc_val + ccc_test} images")
    
    print(f"\nCombined:")
    print(f"  Train: {hq_train + ccc_train} images")
    print(f"  Val:   {hq_val + ccc_val} images")
    print(f"  Test:  {hq_test + ccc_test} images")
    print(f"  Total: {hq_train + hq_val + hq_test + ccc_train + ccc_val + ccc_test} images")
    
    # 每字统计
    all_chars = set()
    for split in ['train', 'val', 'test']:
        d = HQ_OUT / split
        if d.exists():
            for sub in d.iterdir():
                if sub.is_dir():
                    all_chars.add(sub.name)
    
    print(f"\nTotal unique chars: {len(all_chars)}")
    
    # 统计 train 每字数量
    train_dir = HQ_OUT / 'train'
    train_counts = {}
    for sub in train_dir.iterdir():
        if sub.is_dir():
            train_counts[sub.name] = len(list(sub.iterdir()))
    
    counts = list(train_counts.values())
    print(f"\nTrain per-char distribution:")
    print(f"  Min: {min(counts)}")
    print(f"  Max: {max(counts)}")
    print(f"  Avg: {sum(counts)/len(counts):.1f}")
    print(f"  Median: {sorted(counts)[len(counts)//2]}")
    
    # 保存报告
    report = f"""Final Dataset Report
====================
HQ:
  Train: {hq_train}
  Val:   {hq_val}
  Test:  {hq_test}

CCC_only:
  Train: {ccc_train}
  Val:   {ccc_val}
  Test:  {ccc_test}

Combined:
  Train: {hq_train + ccc_train}
  Val:   {hq_val + ccc_val}
  Test:  {hq_test + ccc_test}
  Total: {hq_train + hq_val + hq_test + ccc_train + ccc_val + ccc_test}

Unique chars: {len(all_chars)}

Train per-char:
  Min: {min(counts)}
  Max: {max(counts)}
  Avg: {sum(counts)/len(counts):.1f}
  Median: {sorted(counts)[len(counts)//2]}
"""
    report_path = Path('/root/sj-tmp/datasets/final_dataset_report.txt')
    report_path.write_text(report, encoding='utf-8')
    print(f"\nReport saved to: {report_path}")

if __name__ == '__main__':
    main()
