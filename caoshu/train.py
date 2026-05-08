"""
CaoshuReader — train.py
120G硬盘优化版：每1000步保存，保留20个，自动保存最佳模型，带磁盘空间保护
"""

import os
import sys

# 强制行缓冲，确保后台运行（nohup + 重定向）时日志实时刷新
if not sys.stdout.isatty():
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

import glob
import re
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

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


def alignment_loss(pred, target_embed, label_smoothing=0.0):
    """
    pred: (B, num_learns, D) - num_learns个query的输出
    target_embed: (B, D) - 目标字符embedding
    label_smoothing: label smoothing系数（论文：0.1）

    改进：每个 query 都学习目标，但允许它们从不同角度学习
    """
    num_learns = pred.shape[1]
    # 扩展 target 到 num_learns 个副本
    tgt = target_embed.unsqueeze(1).expand(-1, num_learns, -1)  # (B, num_learns, D)

    # 归一化
    pred_norm = F.normalize(pred, dim=-1)
    tgt_norm = F.normalize(tgt, dim=-1)

    # 每个 query 的 cosine similarity
    cosine_sim = (pred_norm * tgt_norm).sum(dim=-1)  # (B, num_learns)

    # Label smoothing: (1 - smoothing) * target + smoothing * uniform
    # 对于cosine similarity，uniform target是0（随机方向）
    if label_smoothing > 0:
        cosine_sim = cosine_sim * (1 - label_smoothing)

    # 对所有 query 求平均 loss
    return (1 - cosine_sim).mean()


@torch.no_grad()
def validate(resampler, val_loader, vit, mlp1, tok_embeddings, tokenizer, dataset, device):
    """在验证集上计算平均loss"""
    resampler.eval()
    total_loss, count = 0.0, 0
    for imgs, labels in val_loader:
        imgs = imgs.to(device).to(torch.bfloat16)
        vit_feats = get_visual_embed(imgs, vit, mlp1)
        pred = resampler(vit_feats)
        chars = [dataset.idx2char[l.item()] for l in labels]
        token_ids = tokenizer(chars, return_tensors='pt', add_special_tokens=False,
                              padding=True, truncation=True, max_length=4).input_ids[:, 0].to(device)
        tgt_embed = tok_embeddings(token_ids)
        total_loss += alignment_loss(pred, tgt_embed).item()
        count += 1
    resampler.train()
    return total_loss / count if count > 0 else float('inf')


def save_best_val_ckpts(save_dir, ckpt_data, val_loss, step, keep_best, best_val_ckpts):
    """保存top-K最佳验证checkpoint"""
    best_val_ckpts.append((val_loss, step))
    best_val_ckpts.sort(key=lambda x: x[0])
    if len(best_val_ckpts) > keep_best:
        best_val_ckpts.pop()

    for rank, (loss, s) in enumerate(best_val_ckpts, 1):
        if s == step:
            path = os.path.join(save_dir, f'caoshu_best_val_{rank}.pt')
            ckpt_data['val_loss'] = val_loss
            torch.save(ckpt_data, path)
            print(f"  🌟 Saved best_val_{rank}: val_loss={val_loss:.4f}")
            break


def check_disk_space_detailed(path):
    """详细磁盘空间检查，返回状态和剩余GB"""
    free_gb, total_gb = get_disk_usage(path)
    if free_gb > 20:
        status = 'normal'
    elif free_gb > 10:
        status = 'warning'
    elif free_gb > 5:
        status = 'critical'
    else:
        status = 'emergency'
    return status, free_gb


def get_disk_usage(path):
    """获取路径所在磁盘的剩余空间（GB）"""
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    return free_gb, total_gb


def cleanup_old_ckpts(save_dir, keep=20):
    """
    智能清理旧checkpoint：
    1. 只删除 caoshu_step*.pt 文件，保留 best 和 final
    2. 按step数字排序（而非文件时间），避免误删
    3. 保留最新的 keep 个
    """
    # 获取所有 step checkpoint
    pattern = os.path.join(save_dir, 'caoshu_step*.pt')
    ckpts = glob.glob(pattern)
    
    if len(ckpts) <= keep:
        return
    
    # 提取step数字并排序
    def extract_step(path):
        match = re.search(r'step(\d+)\.pt$', os.path.basename(path))
        return int(match.group(1)) if match else 0
    
    ckpts_sorted = sorted(ckpts, key=extract_step)
    
    # 删除旧的，保留最新的 keep 个
    to_delete = ckpts_sorted[:-keep]
    deleted_size = 0
    
    for old_path in to_delete:
        try:
            size = os.path.getsize(old_path) / (1024**3)  # GB
            os.remove(old_path)
            deleted_size += size
            print(f"  🗑️  deleted: {os.path.basename(old_path)} ({size:.1f}GB)")
        except Exception as e:
            print(f"  ⚠️  failed to delete {old_path}: {e}")
    
    if deleted_size > 0:
        print(f"  💾 Freed {deleted_size:.1f}GB disk space")


def check_disk_space(save_dir, keep_ckpts, ckpt_size_gb=3.2):
    """检查磁盘空间是否足够"""
    free_gb, total_gb = get_disk_usage(save_dir)
    needed_gb = keep_ckpts * ckpt_size_gb + 20  # 20GB buffer for system
    
    print(f"\n💽 Disk Check: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
    print(f"   Estimated need: {keep_ckpts} ckpts × {ckpt_size_gb}GB + 20GB buffer = {needed_gb:.1f}GB")
    
    if free_gb < needed_gb:
        print(f"   ⚠️  WARNING: Low disk space! Recommend keep_ckpts <= {int((free_gb - 20) / ckpt_size_gb)}")
        return False
    else:
        print(f"   ✅ Disk space sufficient")
        return True


def get_args():
    p = argparse.ArgumentParser(description='CaoshuReader Training - Auto Val/LR/Disk')
    p.add_argument('--data_root', type=str,
        default='/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset')
    p.add_argument('--split', type=str, default='Training')
    p.add_argument('--save_dir', type=str, default='/root/sj-tmp/checkpoints/CaoshuReader')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--grad_accum', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--total_steps', type=int, default=100000)
    p.add_argument('--warmup_steps', type=int, default=10000, help='线性warmup步数（论文建议10k）')
    p.add_argument('--log_every', type=int, default=10)
    p.add_argument('--save_every', type=int, default=5000)
    p.add_argument('--keep_ckpts', type=int, default=5, help='保留最近N个step checkpoint（默认5，约16GB）')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--num_layers', type=int, default=8, help='Resampler层数（论文：8）')
    p.add_argument('--num_learns', type=int, default=12, help='Query数量（论文：12）')
    p.add_argument('--dropout', type=float, default=0.1, help='Dropout率（论文：0.1）')
    p.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing（论文：0.1）')
    p.add_argument('--skip_disk_check', action='store_true', help='跳过磁盘空间检查')
    # 验证与自动LR
    p.add_argument('--val_every', type=int, default=2000, help='每N步在验证集上评估一次')
    p.add_argument('--lr_patience', type=int, default=6, help='val loss停滞N次后降低LR')
    p.add_argument('--lr_factor', type=float, default=0.3, help='LR缩减倍数')
    p.add_argument('--min_lr', type=float, default=1e-7, help='LR下限')
    p.add_argument('--keep_best_val', type=int, default=3, help='保留top-K最佳val loss checkpoint')
    p.add_argument('--early_stop', type=int, default=None, help='val loss连续N次无改善则停止（默认不启用）')
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[CaoshuReader] device={device}, batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, effective_batch={args.batch_size * args.grad_accum}")
    print(f"Config: total_steps={args.total_steps}, save_every={args.save_every}, "
          f"keep_ckpts={args.keep_ckpts} (~{args.keep_ckpts * 3.2:.0f}GB for checkpoints)")
    print(f"Save dir: {args.save_dir}")

    # 磁盘空间检查
    if not args.skip_disk_check:
        check_disk_space(args.save_dir, args.keep_ckpts)

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
    resampler = load_perceiver_resampler(path=None, num_layers=args.num_layers,
                                         num_learns=args.num_learns, dropout=args.dropout)
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

    # 验证集
    try:
        val_dataset = CaoshuDataset(args.data_root, 'Validation', transform=get_transform('Validation'))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True, drop_last=False)
        print(f"验证集: {len(val_dataset)} 样本")
    except Exception as e:
        print(f"⚠️  无法加载验证集: {e}，跳过验证")
        val_loader = None

    optimizer = AdamW(resampler.parameters(), lr=args.lr, weight_decay=1e-2)

    # Warmup + Cosine annealing（论文：10k步线性warmup）
    if args.warmup_steps > 0:
        warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_steps)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=args.total_steps - args.warmup_steps, eta_min=args.min_lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[args.warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=args.min_lr)

    start_step = 0
    best_loss = float('inf')
    best_step = 0
    best_val_loss = float('inf')
    best_val_ckpts = []   # list of (val_loss, step), sorted ascending
    val_loss_history = []
    lr_reduce_count = 0
    no_improve_count = 0
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        resampler.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt.get('step', 0)
        best_loss = ckpt.get('loss', float('inf'))
        best_step = start_step
        print(f"Resume from step {start_step}, previous best loss: {best_loss:.4f}")
        
        # 重建 optimizer 和 scheduler，避免旧版 PyTorch checkpoint 兼容性问题导致卡死
        # （旧版 optimizer state dict 在新版 PyTorch 下可能引发死锁）
        print("Rebuilding optimizer and scheduler from scratch...")
        optimizer = AdamW(resampler.parameters(), lr=args.lr, weight_decay=1e-2)
        if args.warmup_steps > 0:
            warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_steps)
            cosine_sched = CosineAnnealingLR(optimizer, T_max=args.total_steps - args.warmup_steps, eta_min=args.min_lr)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[args.warmup_steps])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=args.min_lr)
        
        # 手动推进 scheduler 到 resume 的 step（跳过 optimizer.step 警告不影响正确性）
        if start_step > 0:
            for _ in range(start_step):
                scheduler.step()
            print(f"Scheduler advanced to step {start_step}, lr={scheduler.get_last_lr()[0]:.2e}")
        
        # 恢复训练时检查是否会超出step限制
        if start_step >= args.total_steps:
            print(f"⚠️  Warning: resume step ({start_step}) >= total_steps ({args.total_steps})")
            print(f"   Increase --total_steps to continue training")

    step = start_step
    optimizer.zero_grad()
    loss_item = float('inf')  # 初始化，避免未定义错误

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

            loss = alignment_loss(pred, tgt_embed, label_smoothing=args.label_smoothing)
            loss_item = loss.item()
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(resampler.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1

            # 验证与自动LR调整
            if val_loader and step % args.val_every == 0:
                val_loss = validate(resampler, val_loader, vit, mlp1, tok_embeddings, tokenizer, val_dataset, device)
                val_loss_history.append(val_loss)

                # 磁盘检查
                disk_status, free_gb = check_disk_space_detailed('/root/sj-tmp/')
                disk_icon = {'normal': '✅', 'warning': '⚠️', 'critical': '🔴', 'emergency': '🚨'}[disk_status]
                print(f"  [Val] step={step:6d} val_loss={val_loss:.4f} | disk: {disk_icon} {free_gb:.1f}GB")

                # 保存最佳val checkpoint
                ckpt_data = {
                    'step': step,
                    'model_state_dict': resampler.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_item,
                    'val_loss': val_loss,
                }
                if disk_status != 'emergency':
                    save_best_val_ckpts(args.save_dir, ckpt_data, val_loss, step, args.keep_best_val, best_val_ckpts)

                # LR plateau检测
                if len(val_loss_history) >= args.lr_patience:
                    recent_best = min(val_loss_history[-args.lr_patience:])
                    if recent_best >= best_val_loss - 1e-4:
                        current_lr = optimizer.param_groups[0]['lr']
                        new_lr = max(current_lr * args.lr_factor, args.min_lr)
                        if new_lr < current_lr:
                            for pg in optimizer.param_groups:
                                pg['lr'] = new_lr
                            scheduler = CosineAnnealingLR(optimizer, T_max=args.total_steps - step, eta_min=args.min_lr)
                            lr_reduce_count += 1
                            val_loss_history.clear()
                            print(f"  📉 LR reduced: {current_lr:.2e} → {new_lr:.2e} (#{lr_reduce_count})")

                # 更新最佳val loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Early stopping
                if args.early_stop and no_improve_count >= args.early_stop:
                    print(f"  🛑 Early stopping: {no_improve_count} validations without improvement")
                    break

            if step % args.log_every == 0:
                print(f"step={step:6d} | loss={loss_item:.4f} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e} | best={best_loss:.4f}@{best_step}")

            # 每 save_every 步保存常规 checkpoint
            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.save_dir, f'caoshu_step{step}.pt')
                disk_status, free_gb = check_disk_space_detailed('/root/sj-tmp/')

                if disk_status == 'emergency':
                    print(f"  🚨 Skip step{step}: disk emergency ({free_gb:.1f}GB left)")
                elif disk_status == 'critical':
                    print(f"  🔴 Skip step{step}: disk critical ({free_gb:.1f}GB left), only saving best_val")
                else:
                    torch.save({
                        'step': step,
                        'model_state_dict': resampler.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss_item,
                    }, ckpt_path)
                    print(f"  💾 saved: caoshu_step{step}.pt (loss={loss_item:.4f}, "
                          f"disk: {free_gb:.1f}GB free)")
                    
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

    # 最终保存（只有当训练真正执行过才保存）
    if step > start_step:
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
        print(f"最佳Train Loss: caoshu_best.pt (step {best_step}, loss {best_loss:.4f})")
        if best_val_ckpts:
            print(f"最佳Val Loss:   caoshu_best_val_1.pt (step {best_val_ckpts[0][1]}, val_loss {best_val_ckpts[0][0]:.4f})")
        print(f"LR缩减次数: {lr_reduce_count}")
        print(f"保留{args.keep_ckpts}个中间step模型 + {args.keep_best_val}个最佳val模型")
        _, free_gb = check_disk_space_detailed('/root/sj-tmp/')
        print(f"当前磁盘剩余: {free_gb:.1f}GB (/root/sj-tmp/)")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"未执行训练（step {start_step} >= {args.total_steps}）")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
