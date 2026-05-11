"""
CaoshuReader — train.py
120G硬盘优化版：每1000步保存，保留20个，自动保存最佳模型，带磁盘空间保护
"""

import os
import sys
import glob
import re
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

PROJECT_ROOT = '/caoshu'
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


@torch.no_grad()
def evaluate_topk(resampler, vit, mlp1, tok_embeddings, val_loader,
                  all_char_embeds, device, topk=5):
    """在 Validation 集上评估 Top-1 和 Top-K 准确率"""
    resampler.eval()
    top1_correct = topk_correct = total = 0

    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vit_feats = get_visual_embed(imgs, vit, mlp1)
            pred = resampler(vit_feats)

        pred = pred.mean(dim=1).float()
        pred_norm = F.normalize(pred, dim=-1)
        similarities = torch.mm(pred_norm, all_char_embeds.t())
        _, topk_pred = similarities.topk(topk, dim=1)

        top1_correct += (topk_pred[:, 0] == labels).sum().item()
        topk_correct += (topk_pred == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    resampler.train()
    return top1_correct / total, topk_correct / total


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
    p = argparse.ArgumentParser(description='CaoshuReader Training - 120GB Disk Optimized')
    p.add_argument('--data_root', type=str,
        default='/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset')
    p.add_argument('--split', type=str, default='Training')
    p.add_argument('--save_dir', type=str, default='/root/sj-tmp/checkpoints/CaoshuReader')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--grad_accum', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--total_steps', type=int, default=100000)  # 增加到支持继续训练
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--save_every', type=int, default=1000)
    p.add_argument('--keep_ckpts', type=int, default=20, help='保留最近N个step checkpoint（建议20，约60GB）')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--skip_disk_check', action='store_true', help='跳过磁盘空间检查')
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
    num_layers = args.num_layers
    num_learns = None
    if args.resume:
        ckpt_meta = torch.load(args.resume, map_location='cpu', weights_only=False)
        state_dict = ckpt_meta['model_state_dict']
        layer_indices = set()
        for k in state_dict.keys():
            if k.startswith('layers.'):
                layer_idx = int(k.split('.')[1])
                layer_indices.add(layer_idx)
        num_layers = max(layer_indices) + 1 if layer_indices else args.num_layers
        num_learns = state_dict['learns'].shape[0] if 'learns' in state_dict else None
        print(f"  从 checkpoint 推断: num_layers={num_layers}, num_learns={num_learns}")
    
    resampler = load_perceiver_resampler(path=None, num_layers=num_layers)
    if num_learns is not None and resampler.learns.shape[0] != num_learns:
        resampler.learns = torch.nn.Parameter(
            torch.randn(num_learns, 4096, device=device, dtype=torch.bfloat16)
        )
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
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=5000, T_mult=2, eta_min=1e-7
    )

    start_step = 0
    best_loss = float('inf')
    best_step = 0
    
    # 加载 Validation 数据集和预计算字符 embedding（用于评估）
    val_dataset = CaoshuDataset(args.data_root, 'Validation', transform=get_transform('Validation'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    all_chars = [val_dataset.idx2char[i] for i in range(len(val_dataset.idx2char))]
    all_embeds = []
    for i in range(0, len(all_chars), 100):
        batch_chars = all_chars[i:i+100]
        batch_tokens = tokenizer(batch_chars, return_tensors='pt', add_special_tokens=False,
                               padding=True, truncation=True, max_length=4).input_ids[:, 0].to(device)
        with torch.no_grad():
            batch_embeds = tok_embeddings(batch_tokens)
        all_embeds.append(batch_embeds)
    all_embeds = torch.cat(all_embeds, dim=0)
    all_char_embeds_norm = F.normalize(all_embeds, dim=-1).float()
    print(f"[Val] Precomputed {len(all_chars)} character embeddings for evaluation")

    if args.resume:
        ckpt = ckpt_meta if 'ckpt_meta' in dir() else torch.load(args.resume, map_location='cpu', weights_only=False)
        state_dict = ckpt['model_state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        resampler.load_state_dict(state_dict)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception as e:
                print(f"⚠️  Scheduler state load failed (type mismatch expected): {e}")
                print(f"   Resetting scheduler to step {ckpt.get('step', 0)}")
                # CosineAnnealingWarmRestarts 无法直接从旧 scheduler 恢复，手动推进
                for _ in range(ckpt.get('step', 0)):
                    scheduler.step()
        start_step = ckpt.get('step', 0)
        best_loss = ckpt.get('loss', float('inf'))
        best_step = start_step
        print(f"Resume from step {start_step}, previous best loss: {best_loss:.4f}")
        
        # 恢复训练时检查是否会超出step限制
        if start_step >= args.total_steps:
            print(f"⚠️  Warning: resume step ({start_step}) >= total_steps ({args.total_steps})")
            print(f"   Increase --total_steps to continue training")

    step = start_step
    optimizer.zero_grad()
    loss_item = float('inf')  # 初始化，避免未定义错误
    measured = False

    while step < args.total_steps:
        for imgs, labels in loader:
            if step >= args.total_steps:
                break

            imgs = imgs.to(device)

            if not measured:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # ========== 冻结模块：no_grad + autocast(bf16) ==========
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    vit_feats = get_visual_embed(imgs, vit, mlp1)
            # =========================================================

            # ========== 可训练模块：autocast(bf16) + backward ==========
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
            # =========================================================

            loss_item = loss.item()
            (loss / args.grad_accum).backward()

            if not measured:
                peak = torch.cuda.max_memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"\n[显存实测] 首个 batch 峰值: {peak:.2f} GB / {total:.2f} GB 总显存")
                print(f"           若接近上限，建议降低 --batch_size 或 --grad_accum")
                measured = True

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(resampler.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1

            if step % args.log_every == 0:
                print(f"step={step:6d} | loss={loss_item:.4f} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e} | best={best_loss:.4f}@{best_step}")

            # 每 1000 步评估 Validation 准确率
            if step % 1000 == 0 and step > 0:
                top1_acc, top5_acc = evaluate_topk(
                    resampler, vit, mlp1, tok_embeddings, val_loader,
                    all_char_embeds_norm, device, topk=5
                )
                print(f"  [Val] Top-1: {top1_acc:.2%}, Top-5: {top5_acc:.2%}")

            # 每 save_every 步保存常规 checkpoint
            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.save_dir, f'caoshu_step{step}.pt')
                
                # 检查磁盘空间，紧急情况下跳过保存
                free_gb, _ = get_disk_usage(args.save_dir)
                if free_gb < 5:  # 只剩5GB时紧急跳过
                    print(f"  ⚠️  Skip saving step{step}: disk full ({free_gb:.1f}GB left)")
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
        print(f"最佳模型: caoshu_best.pt (step {best_step}, loss {best_loss:.4f})")
        print(f"保留{args.keep_ckpts}个中间模型，占用约{args.keep_ckpts*3.2:.0f}GB")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"未执行训练（step {start_step} >= {args.total_steps}）")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
