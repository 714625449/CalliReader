"""
CaoshuReader — dataset.py
數據集結構：{root}/{split}/{字+版本號}/{圖片ID}.jpg
標籤：去掉目錄名末尾的數字後綴，例如 哀1 → 哀
"""

import os
import re
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def strip_suffix(dirname: str) -> str:
    """哀1 → 哀，哀12 → 哀，阿 → 阿"""
    return re.sub(r'\d+$', '', dirname)


class CaoshuDataset(Dataset):
    def __init__(self, root: str, split: str = 'val', transform=None, repeat_train: int = 3):
        """
        root   : 數據集根目錄，子目錄為 train/val/test
        split  : 'train' | 'val' | 'test'
        repeat_train : train 集重複採樣倍數（在線增強用）
        """
        self.root = Path(root) / split
        self.transform = transform
        self.split = split

        # 建立 字→index 映射（新結構：{字}/，無數字後綴）
        all_chars = sorted(set(
            d.name
            for d in self.root.iterdir()
            if d.is_dir()
        ))
        self.char2idx = {c: i for i, c in enumerate(all_chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.num_classes = len(all_chars)

        # 收集所有 (圖片路徑, label_index)
        self.samples = []
        for char_dir in sorted(self.root.iterdir()):
            if not char_dir.is_dir():
                continue
            char = char_dir.name
            if char not in self.char2idx:
                continue
            label = self.char2idx[char]
            for img_path in sorted(char_dir.glob('*.jpg')):
                self.samples.append((img_path, label))

        # train 集 ×3 重複採樣（每個 epoch 同張圖不同增強）
        if split == 'train' and repeat_train > 1:
            self.samples = self.samples * repeat_train

        print(f"[CaoshuDataset] split={split}, "
              f"classes={self.num_classes}, samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')  # 灰階→RGB，供 VIT 使用
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transform(split: str):
    """訓練時在線增強（每個 epoch 隨機），驗證/測試只做 normalize"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if split == 'train':
        return transforms.Compose([
            # 圖片已 resize 到 448，這裡不再 resize
            transforms.RandomRotation(degrees=(-8, 8)),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ElasticTransform(alpha=30.0, sigma=4.0),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloader(
    root: str,
    split: str = 'val',
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = None,
    repeat_train: int = 3,
) -> DataLoader:
    if shuffle is None:
        shuffle = (split == 'train')
    dataset = CaoshuDataset(
        root, split,
        transform=get_transform(split),
        repeat_train=repeat_train if split == 'train' else 1
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    ), dataset


# ── 快速測試 ──────────────────────────────────────────────────
if __name__ == '__main__':
    ROOT = '/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset'

    loader, ds = get_dataloader(ROOT, split='Validation', batch_size=8)

    print(f"總類別數: {ds.num_classes}")
    print(f"樣本示例: {ds.samples[:3]}")

    imgs, labels = next(iter(loader))
    print(f"batch shape : {imgs.shape}")   # [8, 3, 96, 96]
    print(f"label sample: {labels}")
    print(f"label→字   : {[ds.idx2char[l.item()] for l in labels]}")