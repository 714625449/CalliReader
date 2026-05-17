#!/usr/bin/env python3
"""Track 2: 跑 50 张整幅书法图 → 收集每个字的 Top-5 + 置信度"""

import sys
import json
from pathlib import Path

sys.path.insert(0, '/caoshu')
from caoshu.pipeline import CalliReaderPipeline

print("[Track2] Loading pipeline...")
pipeline = CalliReaderPipeline(
    yolo_model_path='params/best.pt',
    checkpoint_path='params/callialign_v2.pth',
    data_root='/root/sj-tmp/datasets/CaoshuMerged',
    conf_thres=0.25,
    num_layers=4
)

with open('/tmp/p2b_sample_50.txt') as f:
    images = [l.strip() for l in f if l.strip()]

print(f"[Track2] Processing {len(images)} images...")
all_results = []

for img_name in images:
    img_path = f'/caoshu/imgs/samples/samples/images/{img_name}'
    output_dir = f'/tmp/p2b_outputs/{Path(img_name).stem}'
    
    try:
        result = pipeline.process_image(
            image_path=img_path,
            output_dir=output_dir,
            topk=5,
            save_crops=False,
            debug=False,
            yolo_imgsz=1344
        )
        all_results.append({
            'image': img_name,
            'total_chars': result['total_chars'],
            'chars': result['chars']
        })
        print(f"  [OK] {img_name} -> {result['total_chars']} chars")
    except Exception as e:
        print(f"  [ERR] {img_name}: {e}")

out_path = '/root/sj-tmp/p2b_raw_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n[Track2] Saved to {out_path}")
print(f"[Track2] Total images processed: {len(all_results)}")
total_chars = sum(r['total_chars'] for r in all_results)
print(f"[Track2] Total chars: {total_chars}")
