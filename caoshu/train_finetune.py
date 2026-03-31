"""
CaoshuReader — 微调脚本（从 checkpoint 继续，降低学习率）
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    x = x.reshape(n, new_w, int(1/scale_factor), new_h, int(1/scale_factor), c)
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


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str,
        default='/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset')
    p.add_argument('--split', type=str, default='Training')
    p.add_argument('--save_dir', type=str, default='/root/sj-tmp/checkpoints/CaoshuReader')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--grad_accum', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-5, help='新的学习率（覆盖 checkpoint 中的）')
    p.add_argument('--total_steps', type=int, default=100000)  # 从 35000 训到 100000
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--save_every', type=int, default=1000)  # 更频繁保存
    p.add_argument('--resume', type=str, required=True, help='checkpoint 路径')
    p.add_argument('--reset_lr', action='store_true', default=True, help='重置学习率')
    p.add_argument('--num_layers', type=int, default=4)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[Fine-tune] Resume from: {args.resume}")
    print(f"[Fine-tune] New LR: {args.lr}, Target steps: {args.total_steps}")

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

    # 加载 checkpoint
    print(f"Loading checkpoint from {args.resume}...")
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
    
    resampler.load_state_dict(ckpt['model_state_dict'])
    start_step = ckpt.get('step', 0)
    
    # 优化器：新建或加载
    optimizer = AdamW(resampler.parameters(), lr=args.lr, weight_decay=1e-2)
    
    if not args.reset_lr and 'optimizer_state_dict' in ckpt:
        # 加载旧优化器状态但修改学习率
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # 强制覆盖学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        print(f"Loaded optimizer state, override LR to {args.lr}")
    else:
        print(f"Using fresh optimizer with LR={args.lr}")

    # 学习率调度：Cosine 从当前步数开始
    remaining_steps = args.total_steps - start_step
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining_steps, eta_min=1e-7)
    
    print(f"Resume from step {start_step}, will train {remaining_steps} more steps")
    print(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")

    step = start_step
    optimizer.zero_grad()

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
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(resampler.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1

            if step % args.log_every == 0:
                print(f"step={step:6d} | loss={loss.item():.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.save_dir, f'caoshu_step{step}.pt')
                torch.save({
                    'step': step,
                    'model_state_dict': resampler.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, ckpt_path)
                print(f"  checkpoint saved → {ckpt_path}")

    final_path = os.path.join(args.save_dir, 'caoshu_final_finetuned.pt')
    torch.save({
        'step': step,
        'model_state_dict': resampler.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item(),
    }, final_path)
    print(f"微调完成 → {final_path}")


if __name__ == '__main__':
    main()
