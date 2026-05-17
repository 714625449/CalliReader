#!/usr/bin/env python3
"""在验证集 A（shufazidian 验证作者）上评估当前模型真实基线"""

import sys
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/caoshu')
from models.model import (
    load_vision_model, load_mlp1, load_perceiver_resampler,
    load_normed_tok_embeddings, load_tokenizer
)
from config.configu import DOWNSAMPLE_RATIO, device
from caoshu.dataset import get_transform
from caoshu.pipeline import get_visual_embed
from torch.utils.data import Dataset
from pathlib import Path

class SimpleDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = Path(root) / split
        self.transform = transform
        self.samples = []
        all_chars = set()
        
        for char_dir in sorted(self.root.iterdir()):
            if not char_dir.is_dir():
                continue
            char = char_dir.name
            all_chars.add(char)
            label = None  # 后面统一编码
            for img_path in sorted(char_dir.glob('*.png')) + sorted(char_dir.glob('*.jpg')):
                self.samples.append((img_path, char))
        
        self.char2idx = {c: i for i, c in enumerate(sorted(all_chars))}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.num_classes = len(all_chars)
        
        # 重新编码 label
        self.samples = [(p, self.char2idx[c]) for p, c in self.samples]
        print(f"[SimpleDataset] split={split}, classes={self.num_classes}, samples={len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

device = torch.device('cuda')

# Load model
print("Loading model...")
vit = load_vision_model(location='cpu').to(device).to(torch.bfloat16)
vit.eval()

mlp1 = load_mlp1(DOWNSAMPLE_RATIO, location='cpu').to(device).to(torch.bfloat16)
mlp1.eval()

ckpt = torch.load('/caoshu/params/callialign_v2.pth', map_location='cpu', weights_only=False)
state_dict = ckpt['model_state_dict']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

layer_indices = set()
for k in state_dict.keys():
    if k.startswith('layers.'):
        layer_indices.add(int(k.split('.')[1]))
inferred_num_layers = max(layer_indices) + 1 if layer_indices else 4
inferred_num_learns = state_dict['learns'].shape[0] if 'learns' in state_dict else 3

resampler = load_perceiver_resampler(None, num_layers=inferred_num_layers)
if resampler.learns.shape[0] != inferred_num_learns:
    resampler.learns = torch.nn.Parameter(
        torch.randn(inferred_num_learns, 4096, device=device, dtype=torch.bfloat16)
    )
resampler.load_state_dict(state_dict)
resampler = resampler.to(device).to(torch.bfloat16)
resampler.eval()

tok_embeddings = load_normed_tok_embeddings(location='cpu').to(device).to(torch.bfloat16)
tok_embeddings.eval()

tokenizer = load_tokenizer()

# Freeze
for p in vit.parameters():
    p.requires_grad = False
for p in mlp1.parameters():
    p.requires_grad = False
for p in resampler.parameters():
    p.requires_grad = False

# Load dataset
print("Loading validation set A...")
dataset = SimpleDataset('/root/sj-tmp/datasets/Validation_Real', 'main_block', transform=get_transform('Validation'))
idx2char = dataset.idx2char
num_classes = len(idx2char)
print(f"Val A: {len(dataset)} samples, {num_classes} chars")

# Precompute embeddings
print("Precomputing char embeddings...")
all_chars = [idx2char[i] for i in range(num_classes)]
all_embeds = []
batch_size = 1000
with torch.no_grad():
    for i in range(0, num_classes, batch_size):
        batch = all_chars[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors='pt', add_special_tokens=False,
                          padding=True, truncation=True, max_length=4).input_ids[:, 0].to(device)
        embeds = tok_embeddings(tokens)
        all_embeds.append(embeds)
all_embeds = torch.cat(all_embeds, dim=0)
all_embeds_norm = F.normalize(all_embeds, dim=-1).float()

# Evaluate
print("Evaluating...")
correct_top1 = 0
correct_top5 = 0
total = 0
all_confs = []

with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        img_tensor, label_idx = dataset[i]
        img_tensor = img_tensor.unsqueeze(0).to(device).to(torch.bfloat16)
        
        vit_out = get_visual_embed(img_tensor, vit, mlp1)
        pred = resampler(vit_out).mean(dim=1).float()
        pred_norm = F.normalize(pred, dim=-1)
        
        similarities = torch.mm(pred_norm, all_embeds_norm.t())
        top5_vals, top5_idx = torch.topk(similarities[0], k=5, dim=-1)
        top1_idx = top5_idx[0].item()
        
        probs = F.softmax(top5_vals, dim=-1)
        conf = probs[0].item()
        all_confs.append(conf)
        
        if top1_idx == label_idx:
            correct_top1 += 1
        if label_idx in top5_idx.cpu().numpy():
            correct_top5 += 1
        total += 1

top1 = correct_top1 / total * 100
top5 = correct_top5 / total * 100
avg_conf = sum(all_confs) / len(all_confs)

print(f"\n{'='*60}")
print(f"验证集 A (shufazidian 真迹) 评估结果")
print(f"{'='*60}")
print(f"Top-1: {correct_top1}/{total} = {top1:.2f}%")
print(f"Top-5: {correct_top5}/{total} = {top5:.2f}%")
print(f"平均置信度: {avg_conf:.4f}")
print(f"{'='*60}")
