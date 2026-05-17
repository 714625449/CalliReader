#!/usr/bin/env python3
"""
用原始 CalliReader (完整 InternVL VLM) 批量测试 samples 图片
"""
import os
import sys
import json
import Levenshtein
from pathlib import Path
from PIL import Image

sys.path.insert(0, '/caoshu')

import torch
from transformers import AutoModel, AutoTokenizer
from ultralytics import YOLO
import opencc

from utils.utils import calculate_metrics
from config.configu import INTERNVL_PATH, YOLO_CHECKPOINT

# 配置
SAMPLES_DIR = '/caoshu/train/samples'
OUTPUT_DIR = '/caoshu/outputs/samples_test_callireader'
NUM_TEST = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取 samples.json
with open(os.path.join(SAMPLES_DIR, 'samples.json'), 'r', encoding='utf-8') as f:
    samples = json.load(f)

print(f"[Test] 总样本: {len(samples)}, 本次测试: {NUM_TEST}")

# 加载模型（只加载一次）
print("[1/3] 加载 InternVL 模型...")
model = AutoModel.from_pretrained(
    INTERNVL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval().cuda()

print("[2/3] 加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True)

print("[3/3] 加载 YOLO 检测模型...")
detect_model = YOLO(YOLO_CHECKPOINT)

cc = opencc.OpenCC('t2s')

generation_config = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)

results = []
for idx, sample in enumerate(samples[:NUM_TEST]):
    img_name = Path(sample['image']).name
    img_path = os.path.join(SAMPLES_DIR, 'images', img_name)
    gt_text = sample['conversations'][1]['value']
    
    print(f"\n[{idx+1}/{NUM_TEST}] {img_name}")
    print(f"  GT: {gt_text}")
    
    if not os.path.exists(img_path):
        print(f"  ⚠️ 图片不存在，跳过")
        continue
    
    try:
        response, history = model.chat_ocr(
            tokenizer,
            detect_model,
            img_path,
            "这幅书法作品内容是什么？",
            generation_config,
            use_p=True,
            hard_vq=False,
            drop_zero=False,
            repetition_penalty=1.0,
            return_history=True,
            verbose=False
        )
        pred_text = cc.convert(response)
        print(f"  Pred: {pred_text}")
        
        # 计算指标
        pred_chars = list(pred_text)
        gt_chars = list(gt_text)
        precision, recall, f1 = calculate_metrics(pred_chars, gt_chars)
        
        distance = Levenshtein.distance(pred_text, gt_text)
        max_len = max(len(pred_text), len(gt_text))
        ned = distance / max_len if max_len > 0 else 0
        
        print(f"  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, NED={ned:.3f}")
        
        results.append({
            'image': img_name,
            'gt': gt_text,
            'pred': pred_text,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ned': ned,
        })
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()

# 汇总
if results:
    avg_p = sum(r['precision'] for r in results) / len(results)
    avg_r = sum(r['recall'] for r in results) / len(results)
    avg_f1 = sum(r['f1'] for r in results) / len(results)
    avg_ned = sum(r['ned'] for r in results) / len(results)
    
    print(f"\n{'='*60}")
    print(f"汇总结果 (n={len(results)})")
    print(f"  Avg Precision: {avg_p:.3f}")
    print(f"  Avg Recall:    {avg_r:.3f}")
    print(f"  Avg F1:        {avg_f1:.3f}")
    print(f"  Avg NED:       {avg_ned:.3f}")
    print(f"{'='*60}")
    
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'num_tested': len(results),
            'avg_precision': avg_p,
            'avg_recall': avg_r,
            'avg_f1': avg_f1,
            'avg_ned': avg_ned,
            'details': results,
        }, f, ensure_ascii=False, indent=2)
