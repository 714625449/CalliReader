#!/usr/bin/env python3
"""Track 1: 建主验证集 B — 100张整图 YOLO切字 + 生成标注模板"""

import sys
import json
from pathlib import Path

sys.path.insert(0, '/caoshu')
from caoshu.pipeline import CalliReaderPipeline

print("[ValB] Loading pipeline...")
pipeline = CalliReaderPipeline(
    yolo_model_path='params/best.pt',
    checkpoint_path='params/callialign_v2.pth',
    data_root='/root/sj-tmp/datasets/CaoshuMerged',
    conf_thres=0.25,
    num_layers=4
)

val_b_root = Path('/root/sj-tmp/datasets/Validation_Real/main_end2end')
images = sorted([f.name for f in val_b_root.glob('*.jpg')])
print(f"[ValB] Processing {len(images)} images...")

all_results = []

for img_name in images:
    img_path = val_b_root / img_name
    output_dir = val_b_root / 'crops' / Path(img_name).stem
    
    try:
        result = pipeline.process_image(
            image_path=str(img_path),
            output_dir=str(output_dir),
            topk=5,
            save_crops=True,
            debug=False,
            yolo_imgsz=1344
        )
        
        # 简化数据：只保留关键信息
        chars = []
        for c in result['chars']:
            chars.append({
                'id': c['id'],
                'bbox': c['bbox'],
                'model_pred': c['best_char'],
                'confidence': c['best_confidence'],
                'top5': [x['char'] for x in c['top_candidates']],
                'ground_truth': ''  # 待人工填写
            })
        
        all_results.append({
            'image': img_name,
            'total_chars': result['total_chars'],
            'chars': chars
        })
        
        print(f"  [OK] {img_name} -> {result['total_chars']} chars")
    except Exception as e:
        print(f"  [ERR] {img_name}: {e}")

# 保存标注模板
template = {
    'description': '端到端验证集 B 人工标注模板',
    'instructions': '对每个字块，查看 crops/{图片名}/char_XXX.jpg，填入真实标签到 ground_truth 字段',
    'images': all_results
}

with open(val_b_root / 'annotation.json', 'w', encoding='utf-8') as f:
    json.dump(template, f, ensure_ascii=False, indent=2)

total_chars = sum(r['total_chars'] for r in all_results)
print(f"\n[ValB] Done! {len(all_results)} images, {total_chars} chars")
print(f"[ValB] Crops saved to: {val_b_root / 'crops'}")
print(f"[ValB] Annotation template: {val_b_root / 'annotation.json'}")
