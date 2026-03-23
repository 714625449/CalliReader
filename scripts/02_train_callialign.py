#!/usr/bin/env python3
"""
Step 2: 训练 CalliAlign (Perceiver Resampler)
使用 CursiveChineseCalligraphyDataset 的单字数据
针对 RTX 3090 20GB 优化
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
import json
import argparse

sys.path.append('/workspace/CalliReader')

from models.perceiver_resampler import PerceiverResampler
from models.model import load_vision_model, load_mlp1
from config.configu import *
from utils.utils import build_transform

# ============ 配置 ============
class Config:
    # 数据路径
    data_root = "/workspace/CalliReader/data/callireader_cursive"
    output_dir = "/workspace/CalliReader/params"
    
    # 训练参数 (针对 RTX 3090 20GB 优化)
    batch_size = 12          # 单卡20GB可承受
    accumulation_steps = 4   # 有效 batch = 48
    num_epochs = 10
    lr = 1e-4
    weight_decay = 1e-5
    num_workers = 4
    
    # 保存设置
    save_interval = 2000     # 每2000步保存
    eval_interval = 1000     # 每1000步验证
    
    # 混合精度
    mixed_precision = True
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CursiveCharDataset(Dataset):
    """草书单字数据集"""
    def __init__(self, root_dir, char_mapping, transform=None):
        self.root_dir = root_dir
        self.transform = transform or build_transform(input_size=448)
        self.samples = []
        self.char_mapping = char_mapping
        
        print(f"加载数据集: {root_dir}")
        
        # 遍历所有字符文件夹
        char_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        for char_name in tqdm(char_folders, desc="扫描字符"):
            if char_name not in self.char_mapping:
                continue
                
            char_dir = os.path.join(root_dir, char_name)
            char_idx = self.char_mapping[char_name]
            
            for img_name in os.listdir(char_dir):
                if img_name.endswith('.jpg'):
                    self.samples.append({
                        'image': os.path.join(char_dir, img_name),
                        'char_idx': char_idx,
                        'char': char_name
                    })
        
        print(f"加载完成: {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['char_idx'], sample['char']

def create_char_embeddings(tokenizer, text_embed_layer, char_mapping, device):
    """为所有字符创建归一化的文本嵌入目标"""
    print("预计算字符嵌入目标...")
    
    char_targets = {}
    
    # 加载归一化参数
    mu_sigma = torch.load(NORM_PARAMS_PATH)
    mu = mu_sigma['weight'][:, 0].reshape((-1, 1)).to(device)
    sigma = mu_sigma['weight'][:, 1].reshape((-1, 1)).to(device)
    
    with torch.no_grad():
        for char, idx in tqdm(char_mapping.items(), desc="生成嵌入"):
            try:
                # 获取token ID
                tokens = tokenizer(char, return_tensors='pt', add_special_tokens=False)
                token_id = tokens['input_ids'][0, 0].item()
                
                # 获取嵌入
                embed = text_embed_layer(torch.tensor([[token_id]]).to(device))
                
                # Layer normalization
                mean = embed.mean(dim=-1, keepdim=True)
                std = embed.std(dim=-1, keepdim=True)
                embed_norm = (embed - mean) / (std + 1e-6)
                
                char_targets[idx] = embed_norm.squeeze(0).squeeze(0).cpu()
            except Exception as e:
                # 使用零向量作为fallback
                char_targets[idx] = torch.zeros(4096)
    
    return char_targets

def train_epoch(model, train_loader, char_targets, vit, mlp1, optimizer, scheduler, criterion, scaler, config, epoch):
    """训练一个epoch"""
    model.train()
    
    epoch_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    for batch_idx, (images, char_idxs, chars) in enumerate(pbar):
        images = images.to(config.device).to(torch.bfloat16)
        
        # 构建目标
        targets = torch.stack([char_targets[idx.item()] for idx in char_idxs])
        targets = targets.to(config.device).to(torch.bfloat16)
        targets = targets.unsqueeze(1).expand(-1, 3, -1)  # (B, 3, 4096)
        
        # 前向传播
        with autocast(enabled=config.mixed_precision):
            # 1. ViT 特征提取 (冻结)
            with torch.no_grad():
                vit_embeds = vit(
                    pixel_values=images,
                    output_hidden_states=False,
                    return_dict=True
                ).last_hidden_state[:, 1:, :]  # 去掉CLS token
                
                # Reshape 和 pixel shuffle
                B = images.size(0)
                h = w = int(vit_embeds.shape[1] ** 0.5)
                vit_embeds = vit_embeds.reshape(B, h, w, -1)
                
                # Downsample (0.5倍)
                vit_embeds = vit_embeds.view(B, h, int(w * 0.5), -1)
                vit_embeds = vit_embeds.permute(0, 2, 1, 3).contiguous()
                vit_embeds = vit_embeds.view(B, int(h * 0.5), int(w * 0.5), -1)
                vit_embeds = vit_embeds.permute(0, 2, 1, 3).contiguous()
                vit_embeds = vit_embeds.reshape(B, -1, vit_embeds.shape[-1])
                
                # MLP1
                vit_embeds = mlp1(vit_embeds)  # (B, 256, 4096)
            
            # 2. Resampler (训练)
            output = model(vit_embeds)  # (B, 3, 4096)
            
            # 3. 计算损失
            loss = criterion(output, targets)
            loss = loss / config.accumulation_steps
        
        # 反向传播
        if config.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度累积更新
        if (batch_idx + 1) % config.accumulation_steps == 0:
            if config.mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        # 记录
        epoch_loss += loss.item() * config.accumulation_steps
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{epoch_loss / num_batches:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return epoch_loss / num_batches

def validate(model, val_loader, char_targets, vit, mlp1, criterion, config):
    """验证"""
    model.eval()
    
    val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, char_idxs, chars in tqdm(val_loader, desc="验证"):
            images = images.to(config.device).to(torch.bfloat16)
            
            targets = torch.stack([char_targets[idx.item()] for idx in char_idxs])
            targets = targets.to(config.device).to(torch.bfloat16)
            targets = targets.unsqueeze(1).expand(-1, 3, -1)
            
            # ViT特征提取
            vit_embeds = vit(pixel_values=images, output_hidden_states=False, return_dict=True).last_hidden_state[:, 1:, :]
            B = images.size(0)
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(B, h, w, -1)
            vit_embeds = vit_embeds.view(B, h, int(w * 0.5), -1)
            vit_embeds = vit_embeds.permute(0, 2, 1, 3).contiguous()
            vit_embeds = vit_embeds.view(B, int(h * 0.5), int(w * 0.5), -1)
            vit_embeds = vit_embeds.permute(0, 2, 1, 3).contiguous()
            vit_embeds = vit_embeds.reshape(B, -1, vit_embeds.shape[-1])
            vit_embeds = mlp1(vit_embeds)
            
            # Resampler
            output = model(vit_embeds)
            
            loss = criterion(output, targets)
            val_loss += loss.item()
            num_batches += 1
    
    return val_loss / num_batches

def main():
    config = Config()
    
    print("=" * 70)
    print("CalliAlign 训练脚本 (草书优化版)")
    print("=" * 70)
    print(f"设备: {config.device}")
    print(f"配置: batch_size={config.batch_size}, accumulation={config.accumulation_steps}")
    print(f"有效batch size: {config.batch_size * config.accumulation_steps}")
    print("=" * 70)
    
    # 检查数据
    char_mapping_path = os.path.join(config.data_root, "char_mapping.json")
    if not os.path.exists(char_mapping_path):
        print(f"错误: 找不到字符映射文件: {char_mapping_path}")
        print("请先运行 01_prepare_cursive_dataset.py")
        return
    
    with open(char_mapping_path, 'r', encoding='utf-8') as f:
        char_mapping = json.load(f)
    
    print(f"字符种类: {len(char_mapping)}")
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # ========== 1. 加载模型组件 ==========
    print("\n[1/4] 加载模型组件...")
    
    device = torch.device(config.device)
    
    # 加载 ViT 和 MLP1 (冻结)
    print("  加载 ViT...")
    vit = load_vision_model().to(device).to(torch.bfloat16)
    for param in vit.parameters():
        param.requires_grad = False
    vit.eval()
    
    print("  加载 MLP1...")
    mlp1 = load_mlp1(downsample_ratio=0.5).to(device).to(torch.bfloat16)
    for param in mlp1.parameters():
        param.requires_grad = False
    mlp1.eval()
    
    # 初始化 Resampler (唯一需要训练的模块)
    print("  初始化 Resampler...")
    resampler = PerceiverResampler(
        dim=4096,
        depth=4,
        num_latents=3,
        dim_head=128,
        heads=8,
        num_time_embeds=1,
    ).to(device).to(torch.bfloat16)
    
    num_params = sum(p.numel() for p in resampler.parameters())
    print(f"  Resampler 参数量: {num_params / 1e6:.2f}M")
    
    # ========== 2. 加载 Tokenizer 和文本嵌入 ==========
    print("\n[2/4] 加载 Tokenizer...")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True)
    
    # 加载文本嵌入层
    text_embed_layer = nn.Embedding(92553, 4096, padding_idx=2).to(device).to(torch.bfloat16)
    text_embed_layer.load_state_dict(torch.load(TOK_EMBEDDING_PATH, weights_only=True))
    text_embed_layer.eval()
    for param in text_embed_layer.parameters():
        param.requires_grad = False
    
    # 创建字符目标嵌入
    char_targets = create_char_embeddings(tokenizer, text_embed_layer, char_mapping, device)
    
    # ========== 3. 准备数据 ==========
    print("\n[3/4] 准备数据集...")
    
    train_dataset = CursiveCharDataset(
        os.path.join(config.data_root, "single_char", "train"),
        char_mapping
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # 验证集
    val_dataset = CursiveCharDataset(
        os.path.join(config.data_root, "single_char", "val"),
        char_mapping
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # ========== 4. 训练设置 ==========
    print("\n[4/4] 设置优化器...")
    
    optimizer = optim.AdamW(
        resampler.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    criterion = nn.MSELoss()
    scaler = GradScaler() if config.mixed_precision else None
    
    # ========== 5. 训练循环 ==========
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    min_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(config.num_epochs):
        # 训练
        train_loss = train_epoch(
            resampler, train_loader, char_targets, vit, mlp1,
            optimizer, scheduler, criterion, scaler, config, epoch
        )
        
        # 验证
        val_loss = validate(resampler, val_loader, char_targets, vit, mlp1, criterion, config)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': resampler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'char_mapping': char_mapping,
            }
            torch.save(checkpoint, os.path.join(config.output_dir, 'callialign_cursive_best.pth'))
            print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.4f})")
        
        # 保存epoch检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': resampler.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'char_mapping': char_mapping,
        }
        torch.save(checkpoint, os.path.join(config.output_dir, f'callialign_cursive_epoch{epoch+1}.pth'))
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"最佳验证损失: {min_val_loss:.4f}")
    print(f"模型保存至: {config.output_dir}/callialign_cursive_best.pth")
    print("=" * 70)

if __name__ == "__main__":
    main()
