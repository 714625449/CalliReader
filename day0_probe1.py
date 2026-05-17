#!/usr/bin/env python3
"""Day 0 Probe 1: Paired control - shufazidian 48K vs CCC/Val original 12K"""

import os
import sys
import csv
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
from tqdm import tqdm

os.chdir('/caoshu')
sys.path.insert(0, '/caoshu')

from models.model import (
    load_vision_model, load_mlp1, load_perceiver_resampler,
    load_normed_tok_embeddings, load_tokenizer
)
from config.configu import DOWNSAMPLE_RATIO, device
from caoshu.dataset import CaoshuDataset, get_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ========== Load model components ==========
print("Loading model components...")
vit = load_vision_model(location='cpu')
vit = vit.to(device).to(torch.bfloat16)
vit.eval()

mlp1 = load_mlp1(DOWNSAMPLE_RATIO, location='cpu')
mlp1 = mlp1.to(device).to(torch.bfloat16)
mlp1.eval()

# Load checkpoint and infer structure
checkpoint_path = '/caoshu/params/callialign_v2.pth'
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = ckpt['model_state_dict']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

layer_indices = set()
for k in state_dict.keys():
    if k.startswith('layers.'):
        layer_idx = int(k.split('.')[1])
        layer_indices.add(layer_idx)
inferred_num_layers = max(layer_indices) + 1 if layer_indices else 4
inferred_num_learns = state_dict['learns'].shape[0] if 'learns' in state_dict else 3
print(f"Inferred: num_layers={inferred_num_layers}, num_learns={inferred_num_learns}")

resampler = load_perceiver_resampler(None, num_layers=inferred_num_layers)
if resampler.learns.shape[0] != inferred_num_learns:
    resampler.learns = torch.nn.Parameter(
        torch.randn(inferred_num_learns, 4096, device=device, dtype=torch.bfloat16)
    )
resampler.load_state_dict(state_dict)
resampler = resampler.to(device).to(torch.bfloat16)
resampler.eval()

tok_embeddings = load_normed_tok_embeddings(location='cpu')
tok_embeddings = tok_embeddings.to(device).to(torch.bfloat16)
tok_embeddings.eval()

tokenizer = load_tokenizer()

# Freeze all
for p in vit.parameters():
    p.requires_grad = False
for p in mlp1.parameters():
    p.requires_grad = False
for p in resampler.parameters():
    p.requires_grad = False

# ========== Load char mapping ==========
print("Loading char mapping...")
dataset = CaoshuDataset('/root/sj-tmp/datasets/CaoshuMerged', 'Training', transform=None)
idx2char = dataset.idx2char
num_classes = len(idx2char)
print(f"Total chars: {num_classes}")

# ========== Precompute char embeddings ==========
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
all_embeds_norm = F.normalize(all_embeds, dim=-1)

# ========== Transform ==========
val_transform = get_transform('Validation')

# ========== Color check (Probe 2 merged) ==========
print("\n=== Color Space Check ===")
train_sample_path = None
for d in ['一', '不', '之']:
    p = f'/root/sj-tmp/datasets/CaoshuMerged/Training/{d}'
    if os.path.exists(p):
        files = [f for f in os.listdir(p) if f.endswith(('.jpg', '.png')) and '_gen' not in f]
        if files:
            train_sample_path = os.path.join(p, files[0])
            break

shufa_sample_path = None
for a in os.listdir('/root/sj-tmp/datasets/shufazidian/c'):
    p = os.path.join('/root/sj-tmp/datasets/shufazidian/c', a)
    if os.path.isdir(p):
        files = [f for f in os.listdir(p) if f.endswith('.png')]
        if files:
            shufa_sample_path = os.path.join(p, files[0])
            break

train_img = Image.open(train_sample_path).convert('RGB')
shufa_img = Image.open(shufa_sample_path).convert('RGB')
train_mean = np.array(train_img).mean()
shufa_mean = np.array(shufa_img).mean()

print(f"Training sample mean pixel: {train_mean:.1f} (0=black, 255=white)")
print(f"Shufazidian sample mean pixel: {shufa_mean:.1f} (0=black, 255=white)")

needs_invert = shufa_mean > train_mean + 50
print(f"Needs invert: {needs_invert}")

# ========== Inference function ==========
char2idx = {c: i for i, c in idx2char.items()}

def infer_batch(images):
    with torch.no_grad():
        batch = torch.stack(images).to(device).to(torch.bfloat16)
        vit_out = vit(batch).last_hidden_state
        B, N, C = vit_out.shape
        H = W = int(N ** 0.5)
        vit_out = vit_out.transpose(1, 2).reshape(B, C, H, W)
        vit_out = mlp1(vit_out)
        pred = resampler(vit_out).mean(dim=1).float()
        pred_norm = F.normalize(pred, dim=-1)
        similarities = torch.mm(pred_norm, all_embeds_norm.t())
        _, top1_idx = similarities.max(dim=1)
    return top1_idx.cpu().numpy()

# ========== Probe A: shufazidian ==========
print("\n=== Probe A: shufazidian ===")

# Parse CSV, deduplicate
shufa_data = []
seen = set()
with open('/root/sj-tmp/datasets/shufazidian/labels_with_size.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['字'], row['作者'], row['图片标识'])
        if key in seen:
            continue
        seen.add(key)
        pic_id = row['图片标识']
        author_code, img_name = pic_id.split('/')
        img_path = f"/root/sj-tmp/datasets/shufazidian/c/{author_code}/{img_name}.png"
        if os.path.exists(img_path):
            shufa_data.append((img_path, row['字']))

print(f"Unique shufazidian samples: {len(shufa_data)}")

batch_size = 64
shufa_correct = 0
shufa_total = 0
images = []
labels = []

for img_path, char in tqdm(shufa_data, desc="shufazidian"):
    try:
        img = Image.open(img_path).convert('RGB')
        if needs_invert:
            img = ImageOps.invert(img)
        img_tensor = val_transform(img)
        images.append(img_tensor)
        labels.append(char)
        
        if len(images) >= batch_size:
            preds = infer_batch(images)
            for pred_idx, label in zip(preds, labels):
                if idx2char.get(pred_idx) == label:
                    shufa_correct += 1
            shufa_total += len(labels)
            images = []
            labels = []
    except Exception as e:
        continue

if images:
    preds = infer_batch(images)
    for pred_idx, label in zip(preds, labels):
        if idx2char.get(pred_idx) == label:
            shufa_correct += 1
    shufa_total += len(labels)

shufa_top1 = shufa_correct / shufa_total * 100 if shufa_total > 0 else 0
print(f"shufazidian Top-1: {shufa_correct}/{shufa_total} = {shufa_top1:.2f}%")

# ========== Probe B: CCC/Val original ==========
print("\n=== Probe B: CCC_split/Validation original ===")

val_root = '/root/sj-tmp/datasets/CCC_split/Validation'
val_dirs = [d for d in os.listdir(val_root) if os.path.isdir(os.path.join(val_root, d))]

ccc_correct = 0
ccc_total = 0
images = []
labels = []

for d in tqdm(val_dirs, desc="CCC/Val"):
    dir_path = os.path.join(val_root, d)
    files = [f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png', '.jpeg')) and '_gen' not in f]
    for f in files:
        img_path = os.path.join(dir_path, f)
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = val_transform(img)
            images.append(img_tensor)
            labels.append(d)
            
            if len(images) >= batch_size:
                preds = infer_batch(images)
                for pred_idx, label in zip(preds, labels):
                    if idx2char.get(pred_idx) == label:
                        ccc_correct += 1
                ccc_total += len(labels)
                images = []
                labels = []
        except Exception as e:
            continue

if images:
    preds = infer_batch(images)
    for pred_idx, label in zip(preds, labels):
        if idx2char.get(pred_idx) == label:
            ccc_correct += 1
    ccc_total += len(labels)

ccc_top1 = ccc_correct / ccc_total * 100 if ccc_total > 0 else 0
print(f"CCC/Val original Top-1: {ccc_correct}/{ccc_total} = {ccc_top1:.2f}%")

# ========== Summary ==========
print("\n" + "="*60)
print("DAY 0 PROBE 1 RESULTS")
print("="*60)
print(f"shufazidian 48K Top-1:    {shufa_top1:.2f}%")
print(f"CCC/Val original Top-1:   {ccc_top1:.2f}%")
print(f"Difference:               {abs(shufa_top1 - ccc_top1):.2f}%")
print(f"Needs invert:             {needs_invert}")

if abs(shufa_top1 - ccc_top1) < 3:
    conclusion = "Distributions are similar. P1 incremental gain may be limited."
elif shufa_top1 > ccc_top1 + 5:
    conclusion = "shufazidian easier than CCC/Val original. Check preprocessing."
elif shufa_top1 < ccc_top1 - 5:
    conclusion = "shufazidian is a new distribution. P1 priority HIGH."
else:
    conclusion = "Moderate distribution difference. P1 worth trying."

print(f"\n[CONCLUSION] {conclusion}")

# Save findings
with open('/caoshu/day0_findings.md', 'w') as f:
    f.write("# Day 0 Findings\n\n")
    f.write(f"## Probe 1: Paired Control (shufazidian vs CCC/Val original)\n")
    f.write(f"- shufazidian 48K Top-1: {shufa_top1:.2f}% ({shufa_correct}/{shufa_total})\n")
    f.write(f"- CCC/Val original Top-1: {ccc_top1:.2f}% ({ccc_correct}/{ccc_total})\n")
    f.write(f"- Difference: {abs(shufa_top1 - ccc_top1):.2f}%\n")
    f.write(f"- Needs invert: {needs_invert}\n")
    f.write(f"- **Conclusion**: {conclusion}\n")

print("\nSaved to /caoshu/day0_findings.md")
