#!/usr/bin/env python3
"""
多进程版本，使用 OpenCV，支持 resume。
"""
import os
import sys
import random
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np

# 路径
HQ_SOURCE = Path('/root/sj-tmp/datasets/HQ_source')
CCC_SOURCE = Path('/root/sj-tmp/datasets/CCC_only')
HQ_OUT = Path('/root/sj-tmp/datasets/HQ')

TARGET_SIZE = (448, 448)
random.seed(42)

def resize_and_save_cv2(src_path, dst_path):
    """OpenCV resize 并保存"""
    try:
        img = cv2.imread(str(src_path))
        if img is None:
            # 尝试用其他方式读取
            from PIL import Image
            pil_img = Image.open(src_path).convert('RGB')
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(dst_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        print(f"Error {src_path}: {e}")
        return False

def split_files(files):
    """保底划分"""
    random.shuffle(files)
    n = len(files)
    if n == 0:
        return [], [], []
    elif n == 1:
        return files, [], []
    elif n == 2:
        return files[:1], files[1:2], []
    else:
        return files[2:], files[:1], files[1:2]

def process_hq_char(char_name):
    """处理一个 HQ 字"""
    char_dir = HQ_SOURCE / char_name
    
    # 检查是否已处理
    train_dir = HQ_OUT / 'train' / char_name
    if train_dir.exists() and any(train_dir.iterdir()):
        # 已处理，跳过
        return None
    
    files = sorted([f for f in char_dir.iterdir() if f.is_file()])
    files = [f for f in files if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.gif')]
    
    train_files, val_files, test_files = split_files(files)
    
    for split, flist in [('train', train_files), ('val', val_files), ('test', test_files)]:
        if not flist:
            continue
        out_dir = HQ_OUT / split / char_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in flist:
            dst = out_dir / f"{src.stem}.jpg"
            resize_and_save_cv2(src, dst)
    
    return {'char': char_name, 'train': len(train_files), 'val': len(val_files), 'test': len(test_files)}

def process_ccc_char(char_name):
    """处理一个 CCC_only 字"""
    char_dir = CCC_SOURCE / char_name
    
    # 检查是否已处理
    train_dir = HQ_OUT / 'train' / char_name
    if train_dir.exists() and any(f for f in train_dir.iterdir() if '_ccc' in f.name):
        return None
    
    files = sorted([f for f in char_dir.iterdir() if f.is_file()])
    files = [f for f in files if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.gif')]
    
    # 随机抽 20 张
    if len(files) > 20:
        files = random.sample(files, 20)
    
    train_files, val_files, test_files = split_files(files)
    
    for split, flist in [('train', train_files), ('val', val_files), ('test', test_files)]:
        if not flist:
            continue
        out_dir = HQ_OUT / split / char_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in flist:
            dst = out_dir / f"{src.stem}_ccc.jpg"
            resize_and_save_cv2(src, dst)
    
    return {'char': char_name, 'train': len(train_files), 'val': len(val_files), 'test': len(test_files), 'ccc': True}

def main():
    print("Building final dataset (multiprocess + OpenCV)")
    
    # 确保输出目录存在
    for split in ['train', 'val', 'test']:
        (HQ_OUT / split).mkdir(parents=True, exist_ok=True)
    
    # HQ
    hq_chars = sorted([d.name for d in HQ_SOURCE.iterdir() if d.is_dir()])
    print(f"HQ chars: {len(hq_chars)}")
    
    n_workers = min(cpu_count(), 16)
    print(f"Workers: {n_workers}")
    
    with Pool(n_workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(process_hq_char, hq_chars, chunksize=10)):
            if i % 500 == 0:
                print(f"  HQ progress: {i}/{len(hq_chars)}")
            if res:
                results.append(res)
    
    hq_train = sum(r['train'] for r in results)
    hq_val = sum(r['val'] for r in results)
    hq_test = sum(r['test'] for r in results)
    print(f"HQ done: train={hq_train}, val={hq_val}, test={hq_test}")
    
    # CCC_only
    ccc_chars = sorted([d.name for d in CCC_SOURCE.iterdir() if d.is_dir()])
    print(f"\nCCC_only chars: {len(ccc_chars)}")
    
    with Pool(n_workers) as pool:
        results_ccc = []
        for i, res in enumerate(pool.imap_unordered(process_ccc_char, ccc_chars, chunksize=5)):
            if i % 50 == 0:
                print(f"  CCC progress: {i}/{len(ccc_chars)}")
            if res:
                results_ccc.append(res)
    
    ccc_train = sum(r['train'] for r in results_ccc)
    ccc_val = sum(r['val'] for r in results_ccc)
    ccc_test = sum(r['test'] for r in results_ccc)
    print(f"CCC done: train={ccc_train}, val={ccc_val}, test={ccc_test}")
    
    # 统计
    print(f"\n{'='*50}")
    print(f"Combined: train={hq_train+ccc_train}, val={hq_val+ccc_val}, test={hq_test+ccc_test}")
    print(f"Total disk images: {hq_train+hq_val+hq_test+ccc_train+ccc_val+ccc_test}")
    
    # 每字统计
    train_dir = HQ_OUT / 'train'
    counts = [len(list(d.iterdir())) for d in train_dir.iterdir() if d.is_dir()]
    print(f"\nTrain per-char: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}")
    
    # 保存报告
    report = f"""Final Dataset Report
HQ: train={hq_train}, val={hq_val}, test={hq_test}
CCC: train={ccc_train}, val={ccc_val}, test={ccc_test}
Combined: train={hq_train+ccc_train}, val={hq_val+ccc_val}, test={hq_test+ccc_test}
Total: {hq_train+hq_val+hq_test+ccc_train+ccc_val+ccc_test}
Train per-char: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}
"""
    Path('/root/sj-tmp/datasets/final_dataset_report.txt').write_text(report)
    print("Done.")

if __name__ == '__main__':
    main()
