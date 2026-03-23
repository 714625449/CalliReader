#!/usr/bin/env python3
"""
Step 1: 准备草书数据集
将 CursiveChineseCalligraphyDataset 转换为 CalliReader 训练格式
"""

import os
import sys
import json
import shutil
from pathlib import Path
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import numpy as np

# 配置
CCC_DATASET_ROOT = "/root/sj-tmp/CCCdatabase_tmp"
OUTPUT_ROOT = "/workspace/CalliReader/data/callireader_cursive"

def convert_grayscale_to_rgb(src_path, dst_path, target_size=448):
    """将灰度图转换为RGB格式并调整尺寸"""
    try:
        img = Image.open(src_path).convert('L')  # 读取灰度
        
        # 上采样到target_size x target_size
        img = img.resize((target_size, target_size), Image.BICUBIC)
        
        # 转为RGB (复制灰度到三个通道)
        img_rgb = Image.merge('RGB', [img, img, img])
        
        # 确保目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img_rgb.save(dst_path, quality=95)
        return True
    except Exception as e:
        print(f"处理 {src_path} 失败: {e}")
        return False

def prepare_single_char_dataset():
    """
    准备单字数据集 (用于 CalliAlign 预训练)
    输出格式: output/single_char/{train,val,test}/字符名/图片.jpg
    """
    print("=" * 60)
    print("步骤 1: 准备单字数据集")
    print("=" * 60)
    
    if not os.path.exists(CCC_DATASET_ROOT):
        print(f"错误: 数据集不存在: {CCC_DATASET_ROOT}")
        print("请先下载数据集: git clone https://github.com/nccuviplab/CursiveChineseCalligraphyDataset.git")
        return None
    
    splits = ["Training", "Validation", "Test"]
    output_dir = os.path.join(OUTPUT_ROOT, "single_char")
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {"train": 0, "val": 0, "test": 0}
    char_mapping = {}  # 字符到索引的映射
    
    for split in splits:
        split_path = os.path.join(CCC_DATASET_ROOT, f"Cursive_Chinese_Calligraphy_Dataset--{split}")
        if not os.path.exists(split_path):
            print(f"警告: {split_path} 不存在, 跳过")
            continue
        
        # 确定目标split名称
        target_split = split.lower().replace("validation", "val")
        
        print(f"\n处理 {split} -> {target_split}...")
        
        # 遍历每个字符文件夹
        char_folders = [c for c in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, c))]
        
        for char_name in tqdm(char_folders, desc=f"处理字符"):
            char_path = os.path.join(split_path, char_name)
            
            # 记录字符映射
            if char_name not in char_mapping:
                char_mapping[char_name] = len(char_mapping)
            
            # 创建输出目录
            char_output_dir = os.path.join(output_dir, target_split, char_name)
            os.makedirs(char_output_dir, exist_ok=True)
            
            # 处理该字符的所有图片
            img_files = [f for f in os.listdir(char_path) 
                        if f.endswith(('.jpg', '.png', '.bmp', '.jpeg'))]
            
            for img_name in img_files:
                src_img = os.path.join(char_path, img_name)
                dst_img = os.path.join(char_output_dir, img_name.replace('.bmp', '.jpg'))
                
                # 如果已存在则跳过
                if os.path.exists(dst_img):
                    stats[target_split] += 1
                    continue
                
                if convert_grayscale_to_rgb(src_img, dst_img):
                    stats[target_split] += 1
    
    # 保存字符映射
    mapping_path = os.path.join(OUTPUT_ROOT, "char_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(char_mapping, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("单字数据集统计:")
    for split, count in stats.items():
        print(f"  {split}: {count} 张图片")
    print(f"  总计: {sum(stats.values())} 张图片")
    print(f"  字符种类: {len(char_mapping)} 个")
    print(f"  字符映射保存至: {mapping_path}")
    print("=" * 60)
    
    return stats, char_mapping

def synthesize_page_level_dataset(num_pages=5000, chars_per_page_range=(15, 30)):
    """
    合成页面级数据集 (用于 YOLO/OrderFormer/e-IT 训练)
    将单字随机排列合成整页图片，模拟真实书法作品布局
    """
    print("\n" + "=" * 60)
    print(f"步骤 2: 合成页面级数据集 ({num_pages} 页)")
    print("=" * 60)
    
    import numpy as np
    
    output_dir = os.path.join(OUTPUT_ROOT, "page_level")
    images_dir = os.path.join(output_dir, "images")
    annots_dir = os.path.join(output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annots_dir, exist_ok=True)
    
    # 获取训练集所有可用字符和图片
    train_dir = os.path.join(OUTPUT_ROOT, "single_char", "train")
    if not os.path.exists(train_dir):
        print(f"错误: 单字训练集不存在: {train_dir}")
        return
    
    # 构建字符到图片列表的映射
    char_images = {}
    for char_name in os.listdir(train_dir):
        char_dir = os.path.join(train_dir, char_name)
        if not os.path.isdir(char_dir):
            continue
        
        imgs = [os.path.join(char_dir, f) for f in os.listdir(char_dir) 
                if f.endswith('.jpg')]
        if imgs:
            char_images[char_name] = imgs
    
    all_chars = list(char_images.keys())
    print(f"可用字符数: {len(all_chars)}")
    
    # 生成页面
    for page_id in tqdm(range(num_pages), desc="合成页面"):
        # 随机决定页面大小 (模拟不同尺寸的书法作品)
        page_width = random.choice([1024, 1200, 1400])
        page_height = random.choice([1536, 1800, 2000])
        
        # 创建空白页面 (米白色纸张背景)
        page = Image.new('RGB', (page_width, page_height), color=(250, 248, 245))
        
        annotation = {
            "imageHeight": page_height,
            "imageWidth": page_width,
            "shapes": []
        }
        
        # 随机选择字符数量
        num_chars = random.randint(*chars_per_page_range)
        selected_chars = random.sample(all_chars, min(num_chars, len(all_chars)))
        
        # 模拟竖排书法布局: 从右到左，从上到下
        margin_x = 80
        margin_y = 100
        col_width = random.randint(100, 140)
        char_spacing = random.randint(10, 25)
        
        x_pos = page_width - margin_x - col_width  # 从右侧开始
        y_pos = margin_y
        
        char_idx = 0
        for char in selected_chars:
            if char_idx >= num_chars:
                break
            
            # 随机选择该字符的一张图片
            img_path = random.choice(char_images[char])
            
            try:
                # 读取字符图片
                char_img = Image.open(img_path).convert('L')
                
                # 随机调整大小 (模拟不同字号)
                base_size = random.randint(70, 110)
                char_img = char_img.resize((base_size, base_size), Image.BICUBIC)
                
                # 转为RGB
                char_rgb = Image.merge('RGB', [char_img, char_img, char_img])
                
                # 检查是否需要换列
                if y_pos + base_size > page_height - margin_y:
                    x_pos -= col_width
                    y_pos = margin_y
                    if x_pos < margin_x:
                        break  # 页面已满
                
                # 添加随机偏移 (模拟手写自然感)
                offset_x = random.randint(-8, 8)
                offset_y = random.randint(-8, 8)
                
                paste_x = x_pos + offset_x + (col_width - base_size) // 2
                paste_y = y_pos + offset_y
                
                # 粘贴到页面
                page.paste(char_rgb, (paste_x, paste_y))
                
                # 记录标注
                annotation["shapes"].append({
                    "label": char,
                    "points": [
                        [paste_x, paste_y],
                        [paste_x + base_size, paste_y + base_size]
                    ],
                    "turn": char_idx + 1  # 阅读顺序
                })
                
                char_idx += 1
                y_pos += base_size + char_spacing
                
            except Exception as e:
                continue
        
        # 保存页面和标注
        page_filename = f"page_{page_id:06d}.jpg"
        page_path = os.path.join(images_dir, page_filename)
        page.save(page_path, quality=95)
        
        annot_path = os.path.join(annots_dir, f"page_{page_id:06d}.json")
        with open(annot_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
    
    print(f"\n页面级数据集已生成: {num_pages} 页")
    print(f"保存位置: {output_dir}")
    print(f"  图片: {images_dir}")
    print(f"  标注: {annots_dir}")
    
    # 划分训练/验证/测试集
    create_train_val_split(output_dir)

def create_train_val_split(page_dir, ratios=(0.8, 0.1, 0.1)):
    """划分训练/验证/测试集"""
    print("\n划分数据集...")
    
    annots_dir = os.path.join(page_dir, "annotations")
    all_pages = [f.replace('.json', '') for f in os.listdir(annots_dir) if f.endswith('.json')]
    random.shuffle(all_pages)
    
    n_total = len(all_pages)
    n_train = int(n_total * ratios[0])
    n_val = int(n_total * ratios[1])
    
    splits = {
        "train": all_pages[:n_train],
        "val": all_pages[n_train:n_train + n_val],
        "test": all_pages[n_train + n_val:]
    }
    
    for split_name, pages in splits.items():
        split_file = os.path.join(page_dir, f"{split_name}.txt")
        with open(split_file, 'w') as f:
            for page in pages:
                f.write(f"{page}\n")
        print(f"  {split_name}: {len(pages)} 页 -> {split_file}")

def verify_dataset():
    """验证数据集完整性"""
    print("\n" + "=" * 60)
    print("验证数据集")
    print("=" * 60)
    
    # 检查单字数据集
    single_char_dir = os.path.join(OUTPUT_ROOT, "single_char")
    if os.path.exists(single_char_dir):
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(single_char_dir, split)
            if os.path.exists(split_dir):
                char_count = len([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
                img_count = sum(len(os.listdir(os.path.join(split_dir, d))) 
                               for d in os.listdir(split_dir) 
                               if os.path.isdir(os.path.join(split_dir, d)))
                print(f"单字 {split}: {char_count} 个字符, {img_count} 张图片")
    
    # 检查页面级数据集
    page_dir = os.path.join(OUTPUT_ROOT, "page_level")
    if os.path.exists(page_dir):
        for split in ["train", "val", "test"]:
            split_file = os.path.join(page_dir, f"{split}.txt")
            if os.path.exists(split_file):
                with open(split_file) as f:
                    count = len(f.readlines())
                print(f"页面 {split}: {count} 页")
    
    print("=" * 60)

if __name__ == "__main__":
    print("CalliReader 草书数据集准备工具")
    print("=" * 60)
    
    # 1. 准备单字数据集
    result = prepare_single_char_dataset()
    
    if result:
        # 2. 合成页面级数据集
        synthesize_page_level_dataset(num_pages=5000, chars_per_page_range=(15, 30))
        
        # 3. 验证
        verify_dataset()
        
        print("\n✅ 数据准备完成!")
        print(f"输出目录: {OUTPUT_ROOT}")
    else:
        print("\n❌ 数据准备失败，请检查数据集路径")
