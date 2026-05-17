#!/usr/bin/env python3
"""
批量测试 samples 图片
用 pipeline 逻辑做整页识别，和 GT 计算 F1 / NED
"""
import os
import sys
import json
import Levenshtein
from pathlib import Path
from PIL import Image

sys.path.insert(0, '/caoshu')

from caoshu.pipeline import CalliReaderPipeline
from utils.utils import calculate_metrics

# 配置
SAMPLES_DIR = '/caoshu/train/samples'
CKPT = '/caoshu/params/callialign.pth'
DATA_ROOT = '/root/sj-tmp/datasets/CCC_split'
OUTPUT_DIR = '/caoshu/outputs/samples_test'
NUM_TEST = 20  # 先测20张看效果

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取 samples.json
with open(os.path.join(SAMPLES_DIR, 'samples.json'), 'r', encoding='utf-8') as f:
    samples = json.load(f)

print(f"[Test] 总样本: {len(samples)}, 本次测试: {NUM_TEST}")

# 初始化 Pipeline（只初始化一次）
pipeline = CalliReaderPipeline(
    yolo_model_path='/caoshu/params/best.pt',
    checkpoint_path=CKPT,
    data_root=DATA_ROOT,
    conf_thres=0.25,
)

results = []
for idx, sample in enumerate(samples[:NUM_TEST]):
    img_name = Path(sample['image']).name
    img_path = os.path.join(SAMPLES_DIR, 'images', img_name)
    gt_text = sample['conversations'][1]['value']  # GPT 的回答是 GT
    
    print(f"\n[{idx+1}/{NUM_TEST}] {img_name}")
    print(f"  GT: {gt_text}")
    
    if not os.path.exists(img_path):
        print(f"  ⚠️ 图片不存在，跳过")
        continue
    
    try:
        result = pipeline.process_image(
            image_path=img_path,
            output_dir=os.path.join(OUTPUT_DIR, f"sample_{idx}"),
            topk=3,
            save_crops=False,
            yolo_imgsz=1344
        )
        pred_text = result['text']
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
    
    # 保存详细结果
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'num_tested': len(results),
            'avg_precision': avg_p,
            'avg_recall': avg_r,
            'avg_f1': avg_f1,
            'avg_ned': avg_ned,
            'details': results,
        }, f, ensure_ascii=False, indent=2)
