#!/usr/bin/env python3
"""
重新计算 CCC_only，基于已转换为繁体的 HQ。
CCC 数据直接从 CCC_split/Training/ 读取。
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict

HQ_DIR = Path('/root/sj-tmp/datasets/HQ')
CCC_TRAINING_DIR = Path('/root/sj-tmp/datasets/CCC/CCC_split/Training')
CCC_ONLY_DIR = Path('/root/sj-tmp/datasets/CCC_only')

def strip_suffix(name):
    """去掉末尾数字，如 '七2' -> '七', '七' -> '七'"""
    return name.rstrip('0123456789')

def main():
    print("=" * 60)
    print("Recompute CCC_only (HQ is now Traditional)")
    print("=" * 60)
    
    # 1. 读取 HQ 繁体字集
    hq_chars = set()
    for d in HQ_DIR.iterdir():
        if d.is_dir():
            hq_chars.add(d.name)
    print(f"HQ chars (Traditional): {len(hq_chars)}")
    
    # 2. 读取 CCC Training 字集（去数字后缀）
    ccc_chars = set()
    ccc_dirs = {}  # base_char -> [list of full dir names]
    
    for d in sorted(CCC_TRAINING_DIR.iterdir()):
        if d.is_dir():
            base = strip_suffix(d.name)
            ccc_chars.add(base)
            if base not in ccc_dirs:
                ccc_dirs[base] = []
            ccc_dirs[base].append(d.name)
    
    print(f"CCC Training chars: {len(ccc_chars)}")
    
    # 3. 计算 CCC_only
    ccc_only_chars = sorted(ccc_chars - hq_chars)
    print(f"\nCCC_only chars: {len(ccc_only_chars)}")
    
    # 4. 清理并重建 CCC_only
    if CCC_ONLY_DIR.exists():
        print(f"Removing old CCC_only: {CCC_ONLY_DIR}")
        shutil.rmtree(CCC_ONLY_DIR)
    CCC_ONLY_DIR.mkdir(parents=True, exist_ok=True)
    
    total_imgs = 0
    total_dirs = 0
    
    for char in ccc_only_chars:
        # 这个字在 CCC 中的所有目录
        src_names = ccc_dirs.get(char, [])
        if not src_names:
            continue
        
        # 创建目标目录（保留第一个目录名，其他合并进去）
        dest_dir = CCC_ONLY_DIR / src_names[0]
        dest_dir.mkdir(exist_ok=True)
        
        for src_name in src_names:
            src = CCC_TRAINING_DIR / src_name
            if not src.exists():
                continue
            
            # 如果是第一个目录，直接移动
            if src_name == src_names[0]:
                for f in src.iterdir():
                    if f.is_file():
                        shutil.move(str(f), str(dest_dir / f.name))
                        total_imgs += 1
                src.rmdir()
            else:
                # 其他目录，合并进去
                for f in src.iterdir():
                    if f.is_file():
                        dest = dest_dir / f.name
                        if dest.exists():
                            dest = dest_dir / f"{f.stem}_2{f.suffix}"
                        shutil.move(str(f), str(dest))
                        total_imgs += 1
                src.rmdir()
        
        total_dirs += 1
    
    print(f"\nCCC_only rebuilt: {total_dirs} dirs / {total_imgs} images")
    
    # 5. 输出字列表
    print(f"\nAll {len(ccc_only_chars)} CCC_only chars:")
    for i in range(0, len(ccc_only_chars), 50):
        line = ccc_only_chars[i:i+50]
        print(' '.join(line))
    
    # 6. 保存报告
    report = f"""CCC_only Recompute Report
==========================
HQ chars (Traditional): {len(hq_chars)}
CCC Training chars: {len(ccc_chars)}
CCC_only chars: {len(ccc_only_chars)}
CCC_only dirs moved: {total_dirs}
CCC_only images: {total_imgs}

CCC_only chars list ({len(ccc_only_chars)}):
{' '.join(ccc_only_chars)}
"""
    report_path = Path('/root/sj-tmp/datasets/ccc_only_report_v2.txt')
    report_path.write_text(report, encoding='utf-8')
    print(f"\nReport saved to: {report_path}")

if __name__ == '__main__':
    main()
