#!/usr/bin/env python3
"""
将 HQ 数据集的字目录从简体转换为繁体，合并同名目录，
然后重新计算 CCC_only。
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict
import opencc

HQ_DIR = Path('/root/sj-tmp/datasets/HQ')
CCC_RAW_DIR = Path('/root/sj-tmp/datasets/CCC/CCC_split/raw')
CCC_ONLY_DIR = CCC_RAW_DIR / 'CCC_only'

converter = opencc.OpenCC('s2t')

def main():
    print("=" * 60)
    print("Step 1: Convert HQ char dirs from Simplified to Traditional")
    print("=" * 60)
    
    # 先建立一个临时映射
    char_dirs = sorted([d for d in HQ_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(char_dirs)} char dirs in HQ")
    
    # 统计简体->繁体的映射
    s2t_map = {}
    merge_groups = defaultdict(list)  # 繁体 -> [简体目录列表]
    
    for d in char_dirs:
        simp = d.name
        trad = converter.convert(simp)
        s2t_map[simp] = trad
        merge_groups[trad].append(d)
    
    # 找出需要合并的（多个简体映射到同一个繁体）
    merges = {k: v for k, v in merge_groups.items() if len(v) > 1}
    if merges:
        print(f"\nFound {len(merges)} merge groups (multiple simp -> same trad):")
        for trad, dirs in sorted(merges.items()):
            simp_names = [d.name for d in dirs]
            print(f"  {trad} <- {simp_names}")
    
    # Step 1: 创建新的繁体目录，移动/合并文件
    print(f"\nConverting and merging...")
    moved = 0
    merged = 0
    skipped = 0
    
    # 为了安全，先创建一个临时区域
    TEMP_DIR = Path('/root/sj-tmp/datasets/HQ_trad_temp')
    TEMP_DIR.mkdir(exist_ok=True)
    
    for trad, dirs in merge_groups.items():
        trad_dir = TEMP_DIR / trad
        trad_dir.mkdir(exist_ok=True)
        
        for src_dir in dirs:
            for img in src_dir.iterdir():
                if img.is_file():
                    dest = trad_dir / img.name
                    if dest.exists():
                        # 重命名避免冲突
                        base = img.stem
                        suffix = img.suffix
                        dest = trad_dir / f"{base}_from_{src_dir.name}{suffix}"
                        merged += 1
                    shutil.copy2(img, dest)
                    moved += 1
    
    print(f"Copied {moved} images ({merged} renamed due to conflicts)")
    
    # 备份原 HQ，替换为繁体版本
    BACKUP_DIR = Path('/root/sj-tmp/datasets/HQ_simp_backup')
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    shutil.move(HQ_DIR, BACKUP_DIR)
    shutil.move(TEMP_DIR, HQ_DIR)
    print(f"\nOriginal HQ backed up to: {BACKUP_DIR}")
    print(f"New HQ (Traditional) at: {HQ_DIR}")
    
    # 重新统计
    new_dirs = sorted([d for d in HQ_DIR.iterdir() if d.is_dir()])
    new_chars = set(d.name for d in new_dirs)
    print(f"\nAfter conversion: {len(new_dirs)} char dirs")
    
    # Step 2: 重新计算 CCC_only
    print("\n" + "=" * 60)
    print("Step 2: Recompute CCC_only")
    print("=" * 60)
    
    # 清理旧的 CCC_only
    if CCC_ONLY_DIR.exists():
        print(f"Removing old CCC_only: {CCC_ONLY_DIR}")
        shutil.rmtree(CCC_ONLY_DIR)
    
    ccc_chars = set()
    for d in CCC_RAW_DIR.iterdir():
        if d.is_dir() and d.name != 'CCC_only':
            # 去掉数字后缀
            base = d.name.rstrip('0123456789')
            ccc_chars.add(base)
    
    print(f"CCC has {len(ccc_chars)} unique chars")
    
    ccc_only_chars = sorted(ccc_chars - new_chars)
    print(f"CCC_only (not in HQ after trad conversion): {len(ccc_only_chars)} chars")
    
    if ccc_only_chars:
        CCC_ONLY_DIR.mkdir(exist_ok=True)
        for char in ccc_only_chars:
            # 查找 CCC 中这个字的所有目录（含数字后缀）
            for src_dir in CCC_RAW_DIR.iterdir():
                if src_dir.is_dir() and src_dir.name != 'CCC_only':
                    base = src_dir.name.rstrip('0123456789')
                    if base == char:
                        dest = CCC_ONLY_DIR / src_dir.name
                        if dest.exists():
                            # 合并（移动文件）
                            for f in src_dir.iterdir():
                                if f.is_file():
                                    df = dest / f.name
                                    if df.exists():
                                        df = dest / f"{f.stem}_2{f.suffix}"
                                    shutil.copy2(f, df)
                            # 删除源
                            for f in src_dir.iterdir():
                                if f.is_file():
                                    f.unlink()
                            src_dir.rmdir()
                        else:
                            shutil.move(str(src_dir), str(dest))
        
        # 统计
        ccc_only_dirs = [d for d in CCC_ONLY_DIR.iterdir() if d.is_dir()]
        ccc_only_imgs = sum(len(list(d.iterdir())) for d in ccc_only_dirs)
        print(f"\nCCC_only rebuilt: {len(ccc_only_dirs)} dirs / {ccc_only_imgs} images")
    
    # 输出前 100 个 CCC_only 字
    print(f"\nFirst 100 CCC_only chars:")
    for line in [ccc_only_chars[i:i+50] for i in range(0, min(200, len(ccc_only_chars)), 50)]:
        print(' '.join(line))
    
    # 保存完整报告
    report = f"""HQ (Traditional) rebuild report
================================
Original HQ dirs: {len(char_dirs)}
After trad conversion: {len(new_dirs)} dirs
Merge groups: {len(merges)}

CCC chars: {len(ccc_chars)}
CCC_only chars: {len(ccc_only_chars)}
CCC_only dirs: {len([d for d in CCC_ONLY_DIR.iterdir() if d.is_dir()]) if CCC_ONLY_DIR.exists() else 0}
CCC_only images: {sum(len(list(d.iterdir())) for d in CCC_ONLY_DIR.iterdir() if d.is_dir()) if CCC_ONLY_DIR.exists() else 0}

CCC_only chars ({len(ccc_only_chars)}):
{' '.join(ccc_only_chars)}
"""
    report_path = Path('/root/sj-tmp/datasets/hq_trad_report.txt')
    report_path.write_text(report, encoding='utf-8')
    print(f"\nReport saved to: {report_path}")

if __name__ == '__main__':
    main()
