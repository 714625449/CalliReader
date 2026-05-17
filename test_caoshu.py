"""
测试 CaoshuReader checkpoint 在 Validation 集上的准确率
修复：处理 resampler 3D 输出 (B, N, D) -> (B, D)
"""
import os
import sys
import argparse
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader

# 添加项目路径
PROJECT_ROOT = '/caoshu'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'caoshu'))

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

def evaluate(ckpt_path, data_root, split='Validation', num_test=None, batch_size=32):
    device = torch.device('cuda')
    print(f"Testing: {ckpt_path}")
    print(f"Dataset split: {split}")
    
    # 加载组件
    vit = load_vision_model(location='cuda')
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False
        
    mlp1 = load_mlp1(downsample_ratio=DOWNSAMPLE_RATIO)
    mlp1.eval()
    for p in mlp1.parameters():
        p.requires_grad = False
    
    # 从 checkpoint 推断模型结构（兼容不同 num_layers / num_learns）
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict']
    # 移除 DataParallel/DistributedDataParallel 的 module. 前缀
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 推断 num_layers
    layer_indices = set()
    for k in state_dict.keys():
        if k.startswith('layers.'):
            layer_idx = int(k.split('.')[1])
            layer_indices.add(layer_idx)
    num_layers = max(layer_indices) + 1 if layer_indices else 4
    
    # 推断 num_learns
    num_learns = state_dict['learns'].shape[0] if 'learns' in state_dict else 3
    print(f"Inferred model config: num_layers={num_layers}, num_learns={num_learns}")
    
    resampler = load_perceiver_resampler(path=None, num_layers=num_layers)
    # 如果 num_learns 不匹配，需要覆盖
    if resampler.learns.shape[0] != num_learns:
        resampler.learns = torch.nn.Parameter(
            torch.randn(num_learns, 4096, device=device, dtype=torch.bfloat16)
        )
    resampler = resampler.to(device).to(torch.bfloat16)
    
    # 加载 checkpoint
    resampler.load_state_dict(state_dict)
    resampler.eval()
    step = ckpt.get('step', ckpt.get('total_step', '?'))
    loss = ckpt.get('loss', ckpt.get('best_loss', '?'))
    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
    print(f"Loaded checkpoint: step {step}, training loss {loss_str}")
    
    # 加载其他组件
    tok_embeddings, _ = load_normed_tok_embeddings(load_checkboard=True, location='cpu')
    tok_embeddings = tok_embeddings.to(device).to(torch.bfloat16)
    tok_embeddings.eval()
    tokenizer = load_tokenizer()
    
    # 加载数据集
    dataset = CaoshuDataset(data_root, split, transform=get_transform(split))
    print(f"Dataset loaded: {len(dataset)} samples, {len(dataset.idx2char)} classes")
    
    # 随机采样或全量
    if num_test and num_test < len(dataset):
        indices = random.sample(range(len(dataset)), num_test)
        subset = torch.utils.data.Subset(dataset, indices)
        print(f"Testing random subset: {num_test} samples")
    else:
        subset = dataset
        print(f"Testing full Validation set: {len(dataset)} samples")
    
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 预计算所有字符的 embedding
    print("Precomputing character embeddings...")
    all_chars = [dataset.idx2char[i] for i in range(len(dataset.idx2char))]
    all_embeds = []
    for i in range(0, len(all_chars), 100):
        batch_chars = all_chars[i:i+100]
        batch_tokens = tokenizer(batch_chars, return_tensors='pt', add_special_tokens=False,
                               padding=True, truncation=True, max_length=4).input_ids[:, 0].to(device)
        with torch.no_grad():
            batch_embeds = tok_embeddings(batch_tokens)
        all_embeds.append(batch_embeds)
    all_embeds = torch.cat(all_embeds, dim=0)  # (vocab_size, embed_dim)
    all_embeds_norm = torch.nn.functional.normalize(all_embeds, dim=-1).float()  # 转 fp32，与 pred 对齐 dtype
    
    # 评估
    correct = 0
    total = 0
    errors = []
    
    print("Evaluating...")
    for imgs, labels in tqdm(loader):
        imgs = imgs.to(device)
        
        # 前向传播（no_grad + autocast 节省显存）
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                vit_feats = get_visual_embed(imgs, vit, mlp1)
                pred = resampler(vit_feats)  # 可能是 (B, N, D) 或 (B, D)
        
        # autocast 退出后显式转 fp32，避免与 fp32 embedding 做 mm 炸 dtype
        pred = pred.float()
        
        # ==================== 关键修复 ====================
        # PerceiverResampler 输出 (B, num_queries, D)，需要压平到 (B, D)
        if pred.dim() == 3:
            # 对 query 维度做 mean pooling，保留 batch 和 feature 维度
            pred = pred.mean(dim=1)  # (B, N, D) -> (B, D)
        elif pred.dim() != 2:
            # 其他异常情况，强制展平
            B = pred.size(0)
            pred = pred.view(B, -1)
        # ==================================================
        
        # 归一化 (B, D)
        pred_norm = torch.nn.functional.normalize(pred, dim=-1)
        
        # 矩阵乘法计算相似度 (B, D) @ (D, vocab_size) = (B, vocab_size)
        similarities = torch.mm(pred_norm, all_embeds_norm.t())
        pred_ids = similarities.argmax(dim=-1)
        
        # 统计准确率
        for i, (pred_id, true_label) in enumerate(zip(pred_ids, labels)):
            is_correct = (pred_id.item() == true_label.item())
            if is_correct:
                correct += 1
            else:
                if len(errors) < 5:
                    true_char = dataset.idx2char[true_label.item()]
                    pred_char = dataset.idx2char[pred_id.item()]
                    errors.append(f"True: '{true_char}', Pred: '{pred_char}'")
            total += 1
    
    acc = correct / total * 100
    
    print(f"\n{'='*60}")
    print(f"Validation Accuracy: {correct}/{total} = {acc:.2f}%")
    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
    print(f"Checkpoint: step {step}, training loss {loss_str}")
    print(f"{'='*60}")
    
    if errors:
        print(f"\nError examples:")
        for e in errors:
            print(f"  {e}")
    
    # 判断结论
    print(f"\n结论:")
    if acc < 5:
        print("  ❌ 失败：接近随机水平（5301类盲猜约0.02%），建议停止训练")
    elif acc < 15:
        print("  ⚠️  较差：学到少量特征但不稳定，建议停止")
    elif acc < 30:
        print("  ✅ 一般：学到了草书特征，但可能过拟合，建议降低lr继续微调")
    else:
        print("  🌟 良好：模型有效，继续训练")
    
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data_root', type=str, 
                       default='/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset')
    parser.add_argument('--split', type=str, default='Validation')
    parser.add_argument('--num_test', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    evaluate(args.ckpt, args.data_root, args.split, args.num_test, args.batch_size)
