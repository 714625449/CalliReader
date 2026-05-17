#!/usr/bin/env python3
"""
预计算 20 张 samples 的 embedding，验证检测数与 GT 字数是否匹配
"""
import torch
import json
from pathlib import Path
import sys
sys.path.insert(0, '/caoshu')

from transformers import AutoModel
from ultralytics import YOLO


def main():
    print("Loading model...")
    model = AutoModel.from_pretrained(
        '/caoshu/InternVL',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).cuda().eval()

    detect_model = YOLO('/caoshu/params/best.pt')

    with open('/caoshu/train/samples/samples.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)

    output_dir = Path('/caoshu/train/samples_embeddings')
    output_dir.mkdir(exist_ok=True)

    results = []

    for item in samples:
        img_basename = Path(item['image']).name
        img_path = f"/caoshu/train/samples/images/{img_basename}"

        if not Path(img_path).exists():
            print(f"⚠️  Image not found: {img_path}, skipping...")
            continue

        gt_text = item['conversations'][1]['value']

        print(f"\nProcessing {img_basename}...")
        print(f"  GT text: {gt_text} ({len(gt_text)} chars)")

        try:
            with torch.no_grad():
                out_tokens, indices = model.calli_align(
                    img_path=img_path,
                    detect_model=detect_model,
                    drop_zero=False,
                    use_hard_vector_quant=False
                )
        except Exception as e:
            print(f"  ❌ calli_align failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        num_detected = out_tokens.shape[0] // 3
        print(f"  Embedding shape: {out_tokens.shape}")
        print(f"  Detected chars: {num_detected}")
        print(f"  Expected chars: {len(gt_text)}")

        match = (num_detected == len(gt_text))
        status = "✅ MATCH" if match else "❌ MISMATCH"
        print(f"  {status}")

        save_path = output_dir / f"{Path(img_basename).stem}.pt"
        torch.save(out_tokens.cpu(), save_path)

        results.append({
            'image': img_basename,
            'gt_text': gt_text,
            'gt_len': len(gt_text),
            'detected': num_detected,
            'match': match,
            'embedding_path': str(save_path),
            'embedding_shape': list(out_tokens.shape),
            'dtype': str(out_tokens.dtype)
        })

    results_path = output_dir / 'detection_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = len(results)
    matched = sum(1 for r in results if r['match'])
    print(f"\n{'='*60}")
    print(f"Total: {total}, Matched: {matched}, Mismatched: {total - matched}")
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
