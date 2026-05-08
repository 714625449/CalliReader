#!/usr/bin/env python3
"""
整篇书法作品对比评估
对比：原 CalliReader vs 草书 CalliReader
"""
import os
import sys
import json
import re
import Levenshtein

PROJECT_ROOT = '/workspace/CalliReader'
sys.path.insert(0, PROJECT_ROOT)

from config.configu import SEED
from inference import set_seed
from models.model import load_perceiver_resampler
from transformers import AutoModel, AutoTokenizer
from ultralytics import YOLO
import torch
import gc

set_seed(SEED)


def parse_answer_txt(path):
    """解析 answer.txt，提取图片名和正确答案"""
    answers = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_img = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 匹配带图片编号前缀的行，如 "2原文 内容"、"2-1 原文 内容"、"2_2 原文 内容"、"8原文：内容"、"20原文: 内容"
        m = re.match(r'^([\d_\-]+)\s*原文[：:\s]+(.+)$', line)
        if m:
            num, text = m.groups()
            text = text.strip()
            # 统一文件名格式
            img_name = f"{num}.jpg"
            current_img = img_name
            answers[img_name] = text
        elif current_img and not line.startswith('原文') and not line.startswith('识别'):
            # 无前缀的续行，合并到上一张图
            answers[current_img] += line

    return answers


def load_base_model():
    from config.configu import INTERNVL_PATH, YOLO_CHECKPOINT
    print(f"Loading base model from {INTERNVL_PATH}...")
    model = AutoModel.from_pretrained(
        INTERNVL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True)
    detect_model = YOLO(str(YOLO_CHECKPOINT))
    generation_config = dict(num_beams=1, max_new_tokens=256, do_sample=False)
    return model, tokenizer, detect_model, generation_config


def load_caoshu_model(ckpt_path, num_layers=8, num_learns=12, dropout=0.1):
    model, tokenizer, detect_model, generation_config = load_base_model()
    print(f"Loading Caoshu CalliAlign from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    old_resampler = model.resampler
    model.resampler = load_perceiver_resampler(
        path=None, num_layers=num_layers, num_learns=num_learns,
        dropout=dropout, checkpoint=None
    )
    model.resampler.load_state_dict(state_dict, strict=True)
    model.resampler = model.resampler.to(torch.bfloat16).cuda()

    # 关键：不修改 num_image_token
    print("Keep num_image_token = 3")

    del old_resampler
    gc.collect()
    torch.cuda.empty_cache()
    return model, tokenizer, detect_model, generation_config


def clean_text(text):
    """清理文本用于计算 CER"""
    # 移除标点、空格、换行
    text = re.sub(r'[\s，。！？、；：""''（）【】《》\n]', '', text)
    return text


def compute_cer(pred, gt):
    """计算字符错误率 (Character Error Rate)"""
    pred_clean = clean_text(pred)
    gt_clean = clean_text(gt)
    if len(gt_clean) == 0:
        return 1.0
    distance = Levenshtein.distance(pred_clean, gt_clean)
    return distance / len(gt_clean)


def evaluate_image(model, tokenizer, detect_model, generation_config, img_path, prompt="读出图中所有文字"):
    try:
        response, _ = model.chat_ocr(
            tokenizer, detect_model, img_path, prompt, generation_config,
            use_p=True, hard_vq=False, drop_zero=False,
            repetition_penalty=1.5, return_history=True, verbose=False
        )
        return response
    except Exception as e:
        return f"ERROR: {str(e)}"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['original', 'caoshu'], required=True)
    parser.add_argument('--examples_dir', type=str, default='/workspace/CalliReader/examples')
    parser.add_argument('--answer_file', type=str, default='/workspace/CalliReader/examples/answer.txt')
    parser.add_argument('--caoshu_ckpt', type=str,
                        default='/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best_val_1.pt')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    # 解析答案
    answers = parse_answer_txt(args.answer_file)
    print(f"Found {len(answers)} images with ground truth:")
    for k, v in answers.items():
        print(f"  {k}: {v[:40]}...")

    # 加载模型
    if args.mode == 'original':
        model, tokenizer, detect_model, generation_config = load_base_model()
        output_file = args.output or 'results/fullpage_original.json'
    else:
        model, tokenizer, detect_model, generation_config = load_caoshu_model(
            args.caoshu_ckpt, num_layers=8, num_learns=12, dropout=0.1
        )
        output_file = args.output or 'results/fullpage_caoshu.json'

    # 评估
    results = []
    total_cer = 0
    total_chars = 0

    for img_name, gt_text in answers.items():
        img_path = os.path.join(args.examples_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Skip: {img_path} not found")
            continue

        print(f"\nEvaluating: {img_name}")
        pred = evaluate_image(model, tokenizer, detect_model, generation_config, img_path)
        cer = compute_cer(pred, gt_text)
        gt_clean = clean_text(gt_text)

        results.append({
            "image": img_name,
            "gt": gt_text,
            "pred": pred,
            "cer": cer,
            "gt_len": len(gt_clean)
        })

        total_cer += cer * len(gt_clean)
        total_chars += len(gt_clean)

        print(f"  GT: {gt_text[:50]}...")
        print(f"  Pred: {pred[:50]}...")
        print(f"  CER: {cer:.2%}")

    avg_cer = total_cer / total_chars if total_chars > 0 else 1.0

    # 保存
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "mode": args.mode,
            "avg_cer": avg_cer,
            "total_chars": total_chars,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Average CER: {avg_cer:.2%}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
