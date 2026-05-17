#!/usr/bin/env python3
"""
Monkey patch 版 smoke test：验证预计算 embedding 能否通过 generate_ocr 生成正确文本
"""
import torch
import json
from pathlib import Path
import sys
sys.path.insert(0, '/caoshu')

from transformers import AutoModel, AutoTokenizer
from ultralytics import YOLO


def main():
    print("Loading model...")
    model = AutoModel.from_pretrained(
        '/caoshu/InternVL',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained(
        '/caoshu/InternVL',
        trust_remote_code=True
    )

    detect_model = YOLO('/caoshu/params/best.pt')

    with open('/caoshu/train/samples_embeddings/detection_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    matched = [r for r in results if r['match']]
    if not matched:
        print("❌ No matched samples found!")
        return

    # 测试前 3 个匹配的样本
    for sample in matched[:3]:
        img_name = sample['image']
        gt_text = sample['gt_text']
        embed_path = sample['embedding_path']

        print(f"\n{'='*60}")
        print(f"Testing: {img_name}")
        print(f"GT: {gt_text}")

        img_path = f"/caoshu/train/samples/images/{img_name}"
        precomputed_embeds = torch.load(embed_path).cuda().to(torch.bfloat16)
        print(f"Embedding shape: {precomputed_embeds.shape}")

        # Monkey patch：替换 calli_align 为直接返回预计算 embedding
        original_calli_align = model.calli_align

        def patched_calli_align(*args, **kwargs):
            return precomputed_embeds, None

        model.calli_align = patched_calli_align

        try:
            response, _ = model.chat_ocr(
                tokenizer,
                detect_model,
                img_path,
                "读一下图片中书写的文字。",
                dict(num_beams=1, max_new_tokens=1024, do_sample=False),
                use_p=True,
                drop_zero=False,
                hard_vq=False,
                repetition_penalty=1.0,
                return_history=True,
                verbose=False
            )

            print(f"Pred: {response}")
            print(f"Match: {response == gt_text}")

        except Exception as e:
            print(f"❌ chat_ocr failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            model.calli_align = original_calli_align


if __name__ == '__main__':
    main()
