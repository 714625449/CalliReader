#!/usr/bin/env python3
"""
Build e-IT (embedding-based Instruction Tuning) dataset for xtuner LoRA training.
Uses matched samples from detection_results.json where detected chars == GT chars.
"""
import json
from pathlib import Path


def main():
    results_path = Path('/caoshu/train/samples_embeddings/detection_results.json')
    output_path = Path('/caoshu/train/samples_eit.json')

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    matched = [r for r in results if r['match']]
    print(f"Total matched samples: {len(matched)}")

    dataset = []
    for idx, r in enumerate(matched):
        gt_text = r['gt_text']
        num_chars = r['gt_len']
        num_tokens = num_chars * 3  # each char -> 3 [UNUSED_TOKEN_140]

        # Replicate chat_ocr behavior: append N*3 [UNUSED_TOKEN_140] to the question
        question = "读一下图片中书写的文字。" + "[UNUSED_TOKEN_140]" * num_tokens

        item = {
            "id": idx,
            "embedding": r['embedding_path'],
            "image": None,  # precomputed embedding, no need for image
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": gt_text}
            ]
        }
        dataset.append(item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(dataset)} samples to {output_path}")

    # Print a sample for verification
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  GT: {sample['conversations'][1]['value']}")
    print(f"  GT len: {len(sample['conversations'][1]['value'])}")
    print(f"  Question len: {len(sample['conversations'][0]['value'])}")
    print(f"  Num [UNUSED_TOKEN_140]: {sample['conversations'][0]['value'].count('[UNUSED_TOKEN_140]')}")
    print(f"  Embedding: {sample['embedding']}")


if __name__ == '__main__':
    main()
