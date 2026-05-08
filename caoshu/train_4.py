"""
CaoshuReader — train.py
158G硬盘优化版：每1000步保存，保留40个，自动保存最佳模型
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


def cleanup_old_ckpts(save_dir, keep=40):
    """保留最新的 keep 个 checkpoint，删除旧的"""
    ckpts = sorted(glob.glob(os.path.join(save_dir, 'caoshu_step*.pt')), 
                   key=os.path.getmtime)
    if len(ckpts) > keep:
        for old in ckpts[:-keep]:
            try:
                os.remove(old)
                print(f"  🗑️  deleted old: {os.path.basename(old)}")
            except:
                pass


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str,
        default='/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset')
    p.add_argument('--split', type=str, default='Training')
    p.add_argument('--save_dir', type=str, default='/root/sj-tmp/checkpoints/CaoshuReader')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--grad_accum', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--total_steps', type=int, default=40000)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--save_every', type=int, default=1000)  # 每1000步保存
    p.add_argument('--keep_ckpts', type=int, default=40, help='保留最近N个checkpoint')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--num_layers', type=int, default=4)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[CaoshuReader] device={device}, batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, effective_batch={args.batch_size * args.grad_accum}")
    print(f"Config: total_steps={args.total_steps}, save_every={args.save_every}, "
          f"keep_ckpts={args.keep_ckpts} (~{args.keep_ckpts * 3.2:.0f}GB)")
    print(f"Save dir: {args.save_dir}")

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
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=1e-6)

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
                print(f"step={step:6d} | loss={loss_item:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | best={best_loss:.4f}@{best_step}")

            # 每 save_every 步保存常规 checkpoint
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
                
                # 清理旧文件，只保留 keep_ckpts 个
                cleanup_old_ckpts(args.save_dir, args.keep_ckpts)

            # 保存最佳模型（如果当前 loss 更低）
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
    
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"最终模型: {final_path} (step {step}, loss {loss_item:.4f})")
    print(f"最佳模型: caoshu_best.pt (step {best_step}, loss {best_loss:.4f})")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
