#!/usr/bin/env python3
"""
合并 CCC (CursiveChineseCalligraphyDataset) 与书法字典草书数据集

流程:
1. 合并 CCC v1 + v2 → /root/sj-tmp/datasets/CCC/
   - 不区分 Training/Test/Validation，只按字合并
   - 跳过文件名含 "gen" 的生成图片
   - 相同内容(MD5)只保留一份
2. 统一 resize 为 224x224 JPEG
3. 合并 shufazidian/c/ 草书图片
   - 通过 labels_with_size.csv 映射 (目录名/文件名 → 汉字)
   - resize 为 224x224 后与 CCC 去重合并

输出:
  /root/sj-tmp/datasets/CCC/<字>/<图片>.jpg
"""

import csv
import hashlib
import io
import shutil
from pathlib import Path
from PIL import Image

# ==================== 配置 ====================
CCC_SRC_DIRS = [
    Path("/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset"),
    Path("/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset_v2"),
]
SHUFA_DIR = Path("/root/sj-tmp/datasets/shufazidian/c")
SHUFA_CSV = Path("/root/sj-tmp/datasets/shufazidian/labels_with_size.csv")
OUTPUT_DIR = Path("/root/sj-tmp/datasets/CCC")
TARGET_SIZE = (224, 224)
JPEG_QUALITY = 95
# ==============================================


def _build_char_map() -> dict:
    """读取 CSV，建立 目录名/文件名(无扩展名) → 汉字 的映射。"""
    char_map = {}
    if not SHUFA_CSV.exists():
        print(f"[警告] CSV 不存在: {SHUFA_CSV}")
        return char_map
    with open(SHUFA_CSV, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = row["图片标识"]
            if key not in char_map:
                char_map[key] = row["字"]
    print(f"[信息] 加载 shufazidian 映射: {len(char_map)} 条")
    return char_map


def _scan_existing(output_dir: Path) -> set:
    """扫描 output_dir 下已有的 (字, md5) 集合，用于去重。"""
    seen = set()
    for f in output_dir.rglob("*"):
        if not f.is_file():
            continue
        char = f.parent.name
        try:
            md5 = hashlib.md5(f.read_bytes()).hexdigest()
            seen.add((char, md5))
        except Exception:
            pass
    print(f"[信息] CCC 现有唯一(字,md5): {len(seen)}")
    return seen


def _resize_to_bytes(img: Image.Image) -> bytes:
    """将图片 resize 后转为 JPEG bytes。"""
    img = img.convert("RGB")
    if img.size != TARGET_SIZE:
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _write_unique(char_dir: Path, stem: str, data: bytes, seen: set, char: str) -> bool:
    """
    将 data 写入 char_dir，按 (char, md5) 去重。
    若目标目录已有同名文件则自动重命名。
    返回是否实际写入新文件。
    """
    md5 = hashlib.md5(data).hexdigest()
    if (char, md5) in seen:
        return False

    dst = char_dir / f"{stem}.jpg"
    counter = 1
    while dst.exists():
        dst = char_dir / f"{stem}_{counter}.jpg"
        counter += 1

    dst.write_bytes(data)
    seen.add((char, md5))
    return True


def merge_ccc_sources(src_dirs: list, output_dir: Path, seen: set):
    """合并 CCC v1/v2 源目录到 output_dir。"""
    copied = skipped_gen = skipped_dup = 0
    for src in src_dirs:
        if not src.exists():
            print(f"[警告] 源目录不存在: {src}")
            continue
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            if "gen" in f.name.lower():
                skipped_gen += 1
                continue

            char = f.parent.name
            try:
                with Image.open(f) as img:
                    data = _resize_to_bytes(img)
            except Exception as e:
                print(f"[错误] 无法读取 {f}: {e}")
                continue

            char_dir = output_dir / char
            char_dir.mkdir(exist_ok=True)
            if _write_unique(char_dir, f.stem, data, seen, char):
                copied += 1
            else:
                skipped_dup += 1

    print(f"[CCC 合并] 复制: {copied}, 跳过(gen): {skipped_gen}, 跳过(重复): {skipped_dup}")


def merge_shufazidian(src_dir: Path, output_dir: Path, seen: set, char_map: dict):
    """将 shufazidian 草书图片合并到 output_dir。"""
    processed = skipped_dup = skipped_notfound = 0
    errors = []
    for f in src_dir.rglob("*.png"):
        key = f"{f.parent.name}/{f.stem}"
        if key not in char_map:
            skipped_notfound += 1
            continue

        char = char_map[key]
        try:
            with Image.open(f) as img:
                data = _resize_to_bytes(img)
        except Exception as e:
            errors.append((str(f), str(e)))
            continue

        char_dir = output_dir / char
        char_dir.mkdir(exist_ok=True)
        if _write_unique(char_dir, f.stem, data, seen, char):
            processed += 1
        else:
            skipped_dup += 1

    print(f"[shufazidian 合并] 复制: {processed}, 跳过(重复): {skipped_dup}, "
          f"跳过(无映射): {skipped_notfound}, 错误: {len(errors)}")
    for p, e in errors[:5]:
        print(f"  [错误] {p}: {e}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("CCC + shufazidian 草书数据集合并工具")
    print("=" * 60)

    # 1. 加载映射 & 扫描现有文件
    char_map = _build_char_map()
    seen = _scan_existing(OUTPUT_DIR)

    # 2. 合并 CCC
    merge_ccc_sources(CCC_SRC_DIRS, OUTPUT_DIR, seen)

    # 3. 合并 shufazidian
    if SHUFA_DIR.exists():
        merge_shufazidian(SHUFA_DIR, OUTPUT_DIR, seen, char_map)
    else:
        print(f"[跳过] shufazidian 目录不存在: {SHUFA_DIR}")

    # 4. 统计
    total_files = len(list(OUTPUT_DIR.rglob("*.jpg")))
    total_dirs = len([d for d in OUTPUT_DIR.iterdir() if d.is_dir()])
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"总字数: {total_dirs}")
    print(f"总图片数: {total_files}")
    print("=" * 60)


if __name__ == "__main__":
    main()
