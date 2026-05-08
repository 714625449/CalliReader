"""
CaoshuReader — train_max50g.py
根据50GB硬盘最大化保留checkpoint，自动计算可训练步数
"""

import os
import sys
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import shutil

PROJECT_ROOT = '/workspace/CalliReader'
sys.path.insert(0, PROJECT_ROOT)

from models.model import (
    load_vision_model,
    load_mlp1,
    load_perceiver_resampler,
    load_normed_tok_embeddings,
    load_tokenizer,
)
from config.configu import DOWNSAMPLE_RATIO
from dataset import CaoshuDataset, get_transform

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    new_c = int(c / (scale_factor * scale_factor))
    x = x.reshape(n, new_w, 2, new_h, 2, c)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(n, new_w, new_h, new_c)
    return x

@torch.no_grad()
def get_visual_embed(imgs, vit, mlp1):
    vit_out = vit(imgs).last_hidden_state[:, 1:, :]
    h = w = int(vit_out.shape[1] ** 0.5)
    vit_out = vit_out.view(vit_out.shape[0], h, w, -1)
    vit_out = pixel_shuffle(vit_out, scale_factor=DOWNSAMPLE_RATIO)
    vit_out = vit_out.view(vit_out.shape[0], -1, vit_out.shape[-1])
    vit_out = mlp1(vit_out)
    return vit_out

def alignment_loss(pred, target_embed):
    tgt = F.normalize(target_embed, dim=-1).unsqueeze(1)
    pred_norm = F.normalize(pred, dim=-1)
    cosine_sim = (pred_norm * tgt).sum(dim=-1)
    return (1 - cosine_sim).mean()

def get_disk_free_gb(path):
    return shutil.disk_usage(path).free / 1024 / 1024 / 1024

def main():
    # 手动配置区域
    SAVE_DIR = '/root/sj-tmp/checkpoints/CaoshuReader'
    CKPT_SIZE_GB = 3.2  # 每个checkpoint约3.2GB
    SAFETY_MARGIN = 5   # 预留5GB给系统和其他文件
    
    # 计算可用空间
    free_gb = get_disk_free_gb(SAVE_DIR)
    usable_gb = max(0, free_gb - SAFETY_MARGIN)
    max_ckpts = int(usable_gb / CKPT_SIZE_GB)
    
    # 保留：best + final + 常规checkpoints
    keep_regular = max(3, max_ckpts - 2)  # 至少保留3个，减去best和final
    
    # 关键配置：你想每多少步保存？
    # 选项A: 密集保存（每500步），训 keep_regular*500 步
    # 选项B: 稀疏保存（每1000步），训 keep_regular*1000 步（推荐）
    # 选项C: 超稀疏（每2000步），训 keep_regular*2000 步
    
    SAVE_EVERY = 1000  # 推荐每1000步，平衡密度和总长度
    MAX_STEPS = keep_regular * SAVE_EVERY
    
    print("=" * 60)
    print("磁盘空间计算")
    print("=" * 60)
    print(f"可用空间: {free_gb:.1f} GB")
    print(f"安全预留: {SAFETY_MARGIN} GB")
    print(f"可用训练: {usable_gb:.1f} GB")
    print(f"每个ckpt: {CKPT_SIZE_GB} GB")
    print(f"最大数量: {max_ckpts} 个")
    print(f"常规保留: {keep_regular} 个")
    print(f"保存间隔: 每 {SAVE_EVERY} 步")
    print(f"可训练:   0 → {MAX_STEPS} 步 ({MAX_STEPS/1000:.0f}K 步)")
    print(f"历史覆盖: 最近 {keep_regular * SAVE_EVERY} 步")
    print("=" * 60)
    
    # 如果空间不足，警告
    if max_ckpts < 5:
        print("❌ 磁盘空间不足，无法训练！需要至少 20GB")
        return
    
    # 参数配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
        default='/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset')
    parser.add_argument('--split', type=str, default='Training')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--total_steps', type=int, default=MAX_STEPS)  # 自动计算
    parser.add_argument('--save_every', type=int, default=SAVE_EVERY)  # 自动计算
    parser.add_argument('--keep_ckpts', type=int, default=keep_regular) # 自动计算
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num_layers', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n[配置] LR={args.lr}, Steps={args.total_steps}, Save every {args.save_every}")
    
    # 加载模型组件
    print("載入 VIT...")
    vit = load_vision_model(location='cuda')
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False
    
    print("載入 mlp1...")
    mlp1 = load_mlp1(downsample_ratio=DOWNSAMPLE_RATIO)
    mlp1.eval()
    for p in mlp1.parameters():
        p.requires_grad = False
    
    print("載入 PerceiverResampler...")
    resampler = load_perceiver_resampler(path=None, num_layers=args.num_layers)
    resampler = resampler.to(device).to(torch.bfloat16)
    resampler.train()
    
    print("載入 token embeddings...")
    tok_embeddings, _ = load_normed_tok_embeddings(load_checkboard=True, location='cpu')
    tok_embeddings = tok_embeddings.to(device).to(torch.bfloat16)
    tok_embeddings.eval()
    for p in tok_embeddings.parameters():
        p.requires_grad = False
    
    print("載入 tokenizer...")
    tokenizer = load_tokenizer()
    
    dataset = CaoshuDataset(args.data_root, args.split,
                            transform=get_transform(args.split))
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    
    optimizer = AdamW(resampler.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=1e-7)
    
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        resampler.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_step = ckpt.get('step', 0)
        print(f"Resume from step {start_step}")
    
    step = start_step
    optimizer.zero_grad()
    best_loss = float('inf')
    best_step = 0
    
    # 清理旧文件（重新开始时）
    if start_step == 0:
        old_files = glob.glob(os.path.join(args.save_dir, 'caoshu_step*.pt'))
        for f in old_files:
            os.remove(f)
        print(f"清理了 {len(old_files)} 个旧文件")
    
    while step < args.total_steps:
        for imgs, labels in loader:
            if step >= args.total_steps:
                break
            
            imgs = imgs.to(device).to(torch.bfloat16)
            
            vit_feats = get_visual_embed(imgs, vit, mlp1)
            pred = resampler(vit_feats)
            
            chars = [dataset.idx2char[l.item()] for l in labels]
            token_ids = tokenizer(
                chars,
                return_tensors='pt',
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=4,
            ).input_ids[:, 0].to(device)
            
            with torch.no_grad():
                tgt_embed = tok_embeddings(token_ids)
            
            loss = alignment_loss(pred, tgt_embed)
            loss_item = loss.item()
            (loss / args.grad_accum).backward()
            
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(resampler.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            step += 1
            
            if step % args.log_every == 0:
                print(f"step={step:6d} | loss={loss_item:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | best={best_loss:.4f}")
            
            # 保存checkpoint（每N步）
            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.save_dir, f'caoshu_step{step}.pt')
                torch.save({
                    'step': step,
                    'model_state_dict': resampler.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_item,
                }, ckpt_path)
                print(f"  💾 saved: caoshu_step{step}.pt (loss={loss_item:.4f})")
                
                # 清理旧文件，只保留最近的 keep_ckpts 个
                ckpts = sorted(glob.glob(os.path.join(args.save_dir, 'caoshu_step*.pt')), 
                              key=os.path.getmtime)
                if len(ckpts) > args.keep_ckpts:
                    for old in ckpts[:-args.keep_ckpts]:
                        os.remove(old)
                        print(f"    🗑️  removed: {os.path.basename(old)}")
            
            # 保存最佳模型
            if loss_item < best_loss:
                best_loss = loss_item
                best_step = step
                best_path = os.path.join(args.save_dir, 'caoshu_best.pt')
                torch.save({
                    'step': step,
                    'model_state_dict': resampler.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_item,
                }, best_path)
                print(f"  🌟 NEW BEST: step {step}, loss {loss_item:.4f}")
    
    # 最终保存
    final_path = os.path.join(args.save_dir, 'caoshu_final.pt')
    torch.save({
        'step': step,
        'model_state_dict': resampler.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss_item,
    }, final_path)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"总步数: {step}")
    print(f"最佳模型: step {best_step}, loss {best_loss:.4f} → caoshu_best.pt")
    print(f"最终模型: step {step}, loss {loss_item:.4f} → caoshu_final.pt")
    print(f"保留的checkpoints: {args.keep_ckpts} 个（最近 {args.keep_ckpts * args.save_every} 步）")
    print("=" * 60)

if __name__ == '__main__':
    main()
