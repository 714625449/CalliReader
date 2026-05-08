"""
用两个checkpoint对比识别results（含YOLO单字检测）：
  - caoshu_best.pt       (最佳 train loss, step 36510)
  - caoshu_best_val_1.pt (最佳 val loss,   step 14000)
"""
import os
import sys
import glob
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np

PROJECT_ROOT = '/workspace/CalliReader'
sys.path.insert(0, PROJECT_ROOT)

from models.model import (
    load_vision_model, load_mlp1, load_perceiver_resampler,
    load_normed_tok_embeddings, load_tokenizer,
)
from config.configu import DOWNSAMPLE_RATIO
from caoshu.dataset import CaoshuDataset
from caoshu.train import pixel_shuffle, get_visual_embed
from ultralytics import YOLO

import torchvision.transforms as T

# ── 配置 ──────────────────────────────────────────────────────────────
DATA_ROOT = '/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset'
CKPT_A    = '/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best.pt'
CKPT_B    = '/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best_val_1.pt'
YOLO_PATH = '/workspace/CalliReader/params/best.pt'
EXAMPLES_DIR = '/workspace/CalliReader/examples'
TOPK      = 3
CONF      = 0.25
MAX_CHARS = 10   # 每张图最多取前N个字，避免输出过长

# ── 预处理 ─────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── 加载公共组件 ────────────────────────────────────────────────────────
print("加载公共模型组件...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vit  = load_vision_model(location='cuda').eval()
mlp1 = load_mlp1(downsample_ratio=DOWNSAMPLE_RATIO).eval()
for p in list(vit.parameters()) + list(mlp1.parameters()):
    p.requires_grad = False

tok_embeddings, _ = load_normed_tok_embeddings(load_checkboard=True, location='cpu')
tok_embeddings = tok_embeddings.to(device).to(torch.bfloat16).eval()
tokenizer = load_tokenizer()

dataset   = CaoshuDataset(DATA_ROOT, 'Validation', transform=None)
idx2char  = dataset.idx2char
num_classes = len(idx2char)
print(f"字符类别数: {num_classes}")

print("预计算字符 embeddings...")
all_chars = [idx2char[i] for i in range(num_classes)]
all_token_ids = tokenizer(all_chars, return_tensors='pt', add_special_tokens=False,
                          padding=True, truncation=True, max_length=4).input_ids[:, 0].to(device)
with torch.no_grad():
    all_embeds      = tok_embeddings(all_token_ids)
    all_embeds_norm = F.normalize(all_embeds, dim=-1)

print("加载 YOLO...")
yolo = YOLO(YOLO_PATH)

# ── 加载 resampler ─────────────────────────────────────────────────────
def load_resampler(ckpt_path, label):
    resampler = load_perceiver_resampler(path=None, num_layers=4)
    resampler = resampler.to(device).to(torch.bfloat16).eval()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    resampler.load_state_dict(sd, strict=True)
    step = ckpt.get('step', '?')
    val_loss = ckpt.get('val_loss', 'N/A')
    train_loss = ckpt.get('loss', '?')
    print(f"  [{label}] step={step}  train_loss={train_loss:.4f}  val_loss={val_loss if isinstance(val_loss,str) else f'{val_loss:.4f}'}")
    return resampler

print("\n加载 Resampler A (best train loss)...")
resampler_a = load_resampler(CKPT_A, 'A')
print("加载 Resampler B (best val loss)...")
resampler_b = load_resampler(CKPT_B, 'B')

# ── 推理 ───────────────────────────────────────────────────────────────
@torch.no_grad()
def recognize(crop_pil, resampler):
    img_t = transform(crop_pil.convert('RGB')).unsqueeze(0).to(device).to(torch.bfloat16)
    feats = get_visual_embed(img_t, vit, mlp1)
    pred  = resampler(feats)
    pred_norm = F.normalize(pred, dim=-1)
    scores = torch.matmul(pred_norm[0], all_embeds_norm.t()).sum(dim=0)
    vals, idxs = torch.topk(scores, k=TOPK)
    return [(idx2char[i.item()], v.item()) for i, v in zip(idxs, vals)]

# ── YOLO 裁字 ──────────────────────────────────────────────────────────
def yolo_crops(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return [], None
    results = yolo(img_bgr, conf=CONF, verbose=False)
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
        conf = float(box.conf[0])
        cx, cy = (x1+x2)//2, (y1+y2)//2
        boxes.append((cy, cx, x1, y1, x2, y2, conf))
    boxes.sort()  # 按阅读顺序（从上到下，从左到右）

    crops = []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    for cy, cx, x1, y1, x2, y2, conf in boxes:
        pad = 4
        x1c, y1c = max(0,x1-pad), max(0,y1-pad)
        x2c, y2c = min(w,x2+pad), min(h,y2+pad)
        crop = img_rgb[y1c:y2c, x1c:x2c]
        if crop.size > 0:
            crops.append((Image.fromarray(crop), conf))
    return crops, len(boxes)

# ── 主循环 ─────────────────────────────────────────────────────────────
jpg_files = sorted(glob.glob(os.path.join(EXAMPLES_DIR, '*.jpg')) +
                   glob.glob(os.path.join(EXAMPLES_DIR, '*.JPG')))
print(f"\n找到 {len(jpg_files)} 张图片，每张取前 {MAX_CHARS} 个字\n")

total_chars = 0
agree_chars = 0
all_diffs = []

for jpg_path in jpg_files:
    fname = os.path.basename(jpg_path)
    crops, n_detected = yolo_crops(jpg_path)

    if not crops:
        print(f"\n[{fname}] ⚠️  YOLO 未检测到字符")
        continue

    show = crops[:MAX_CHARS]
    print(f"\n{'─'*72}")
    print(f"[{fname}]  检测到 {n_detected} 字，显示前 {len(show)} 个")
    print(f"  {'#':<4} {'Best-Train A':<30} {'Best-Val B':<30} {'一致?'}")
    print(f"  {'─'*68}")

    img_agree = 0
    for i, (crop_pil, yolo_conf) in enumerate(show):
        res_a = recognize(crop_pil, resampler_a)
        res_b = recognize(crop_pil, resampler_b)
        top1_a = res_a[0][0]
        top1_b = res_b[0][0]
        match = "✓" if top1_a == top1_b else "✗"
        if top1_a == top1_b:
            agree_chars += 1
            img_agree   += 1
        else:
            all_diffs.append((fname, i+1, res_a, res_b))
        total_chars += 1

        a_str = ' '.join([f"{c}({s:.2f})" for c, s in res_a])
        b_str = ' '.join([f"{c}({s:.2f})" for c, s in res_b])
        print(f"  {i+1:<4} {a_str:<30} {b_str:<30} {match}")

    print(f"  → 本图一致率: {img_agree}/{len(show)}")

# ── 汇总 ───────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"总计字符: {total_chars}")
print(f"Top-1 一致: {agree_chars}/{total_chars} ({agree_chars/total_chars*100:.1f}%)" if total_chars else "无字符")

if all_diffs:
    print(f"\n── 不一致字符详情 ({len(all_diffs)} 处) ──")
    for fname, idx, res_a, res_b in all_diffs:
        a_str = ' | '.join([f"{c}({s:.3f})" for c, s in res_a])
        b_str = ' | '.join([f"{c}({s:.3f})" for c, s in res_b])
        print(f"  {fname} 第{idx}字:")
        print(f"    A (best-train): {a_str}")
        print(f"    B (best-val):   {b_str}")
