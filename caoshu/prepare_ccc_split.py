"""
CCC 数据集划分工具
将 /root/sj-tmp/datasets/CCC/ 按 8:1:1 划分为 Training/Validation/Test

结构转换:
  输入:  /root/sj-tmp/datasets/CCC/{字+版本号}/{图片}.jpg
  输出:  /root/sj-tmp/datasets/CCC_split/{Training,Validation,Test}/{字+版本号}/{图片}.jpg

要求每字至少 10 张图才进入划分，否则全部归入 Training。
"""

import os
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def strip_suffix(dirname: str) -> str:
    """去掉末尾数字，如 哀1 -> 哀，阿 -> 阿"""
    import re
    return re.sub(r'\d+$', '', dirname)


def prepare_split(src_root: str, dst_root: str, min_samples: int = 10,
                  train_ratio: float = 0.8, val_ratio: float = 0.1,
                  seed: int = 42):
    """
    按字级别划分，保证同一字出现在多个 split 中（如果样本足够）
    """
    random.seed(seed)
    src = Path(src_root)
    dst = Path(dst_root)

    if dst.exists():
        print(f"目标目录已存在，删除: {dst}")
        shutil.rmtree(dst)

    # 收集所有样本: char -> list of (img_path, version_dir_name)
    char_samples = defaultdict(list)
    for char_dir in sorted(src.iterdir()):
        if not char_dir.is_dir():
            continue
        for img_path in char_dir.glob('*.jpg'):
            base_char = strip_suffix(char_dir.name)
            char_samples[base_char].append((img_path, char_dir.name))

    total_chars = len(char_samples)
    total_imgs = sum(len(v) for v in char_samples.values())
    print(f"总字符数: {total_chars}")
    print(f"总图片数: {total_imgs}")

    train_samples, val_samples, test_samples = [], [], []
    skipped_chars = []

    for char, samples in sorted(char_samples.items()):
        n = len(samples)
        if n < min_samples * 2:
            # 样本太少，全部进训练集
            train_samples.extend(samples)
            skipped_chars.append((char, n))
            continue

        # 随机打乱后划分
        random.shuffle(samples)
        n_train = max(int(n * train_ratio), n - 2 * min_samples)
        n_val = max(min_samples, int(n * val_ratio))
        n_test = n - n_train - n_val

        # 保证测试集至少有一点
        if n_test < min_samples // 2:
            n_test = min(min_samples, n - n_train - min_samples)
            n_val = n - n_train - n_test

        train_samples.extend(samples[:n_train])
        val_samples.extend(samples[n_train:n_train + n_val])
        test_samples.extend(samples[n_train + n_val:])

    print(f"\n划分结果:")
    print(f"  Training:   {len(train_samples)} 张 ({len(train_samples)/total_imgs*100:.1f}%)")
    print(f"  Validation: {len(val_samples)} 张 ({len(val_samples)/total_imgs*100:.1f}%)")
    print(f"  Test:       {len(test_samples)} 张 ({len(test_samples)/total_imgs*100:.1f}%)")
    print(f"  未划分字符: {len(skipped_chars)} 个（样本过少，全部进训练集）")

    # 验证字符重叠
    def get_chars(sample_list):
        return set(strip_suffix(Path(p).parent.name) for p, _ in sample_list)

    train_chars = get_chars(train_samples)
    val_chars = get_chars(val_samples)
    test_chars = get_chars(test_samples)
    print(f"\n字符覆盖:")
    print(f"  Training 覆盖:   {len(train_chars)} 字")
    print(f"  Validation 覆盖: {len(val_chars)} 字")
    print(f"  Test 覆盖:       {len(test_chars)} 字")
    print(f"  Train ∩ Val:     {len(train_chars & val_chars)} 字")
    print(f"  Train ∩ Test:    {len(train_chars & test_chars)} 字")

    # 复制文件
    def copy_split(samples, split_name):
        split_dir = dst / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for img_path, version_dir in tqdm(samples, desc=f"Copy {split_name}"):
            dest_dir = split_dir / version_dir
            dest_dir.mkdir(exist_ok=True)
            shutil.copy2(img_path, dest_dir / img_path.name)

    print("\n开始复制文件...")
    copy_split(train_samples, 'Training')
    copy_split(val_samples, 'Validation')
    copy_split(test_samples, 'Test')

    print(f"\n✅ 数据集划分完成: {dst}")
    return dst


def create_symlinks_for_params():
    """创建 params 软链接，使 configu.py 中的相对路径生效"""
    callireader_root = Path('/workspace/CalliReader')
    params_src = Path('/root/sj-tmp/checkpoints/CalliReader/params')
    params_link = callireader_root / 'params'

    if params_link.exists() or params_link.is_symlink():
        print(f"params 链接已存在: {params_link}")
        return

    if not params_src.exists():
        print(f"⚠️  源 params 目录不存在: {params_src}")
        return

    params_link.symlink_to(params_src, target_is_directory=True)
    print(f"✅ 创建软链接: {params_link} -> {params_src}")


if __name__ == '__main__':
    SRC = '/root/sj-tmp/datasets/CCC'
    DST = '/root/sj-tmp/datasets/CCC_split'

    prepare_split(SRC, DST, min_samples=10)
    create_symlinks_for_params()

    print("\n" + "=" * 60)
    print("下一步:")
    print(f"  cd /workspace/CalliReader/caoshu")
    print(f"  python train.py --data_root {DST} ...")
    print("=" * 60)
