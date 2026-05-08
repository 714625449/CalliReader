"""
在 CCC Test 集上评估 CalliReader 的字符识别准确率
对比：原模型 vs 草书训练后的模型
"""
import os
import sys
import json
import torch
import random
import argparse
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = '/workspace/CalliReader'
sys.path.insert(0, PROJECT_ROOT)

from config.configu import SEED
from models.model import load_perceiver_resampler
import gc

random.seed(SEED)

def load_base_model():
    from transformers import AutoModel, AutoTokenizer
    from ultralytics import YOLO
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
    generation_config = dict(num_beams=1, max_new_tokens=8, do_sample=False)
    return model, tokenizer, detect_model, generation_config

def load_caoshu_resampler(caoshu_ckpt_path, num_layers=8, num_learns=12, dropout=0.1):
    print(f"Loading Caoshu CalliAlign from {caoshu_ckpt_path}")
    caoshu_ckpt = torch.load(caoshu_ckpt_path, map_location='cpu', weights_only=False)
    state_dict = caoshu_ckpt['model_state_dict'] if 'model_state_dict' in caoshu_ckpt else caoshu_ckpt
    resampler = load_perceiver_resampler(path=None, num_layers=num_layers, num_learns=num_learns, dropout=dropout, checkpoint=None)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    resampler.load_state_dict(state_dict, strict=True)
    resampler = resampler.to(torch.bfloat16).cuda()
    print(f"Successfully loaded Caoshu CalliAlign (layers={num_layers}, learns={num_learns})")
    return resampler

def evaluate_model(model, tokenizer, detect_model, generation_config, test_dir, max_samples=None, prompt="这幅书法作品中的文字是什么？"):
    """在测试集上评估"""
    # 收集所有测试样本
    samples = []
    for char_dir in sorted(Path(test_dir).iterdir()):
        if not char_dir.is_dir():
            continue
        char = char_dir.name
        # 移除数字后缀（如 "哀1" -> "哀"）
        char_clean = ''.join(c for c in char if not c.isdigit())
        for img_path in char_dir.glob('*.jpg'):
            samples.append((str(img_path), char_clean))
    
    if max_samples and max_samples < len(samples):
        samples = random.sample(samples, max_samples)
    
    print(f"Total test samples: {len(samples)}")
    
    correct = 0
    total = 0
    results = []
    
    for img_path, gt_char in tqdm(samples, desc="Evaluating"):
        try:
            response, _ = model.chat_ocr(
                tokenizer, detect_model, img_path, prompt, generation_config,
                use_p=True, hard_vq=False, drop_zero=False,
                repetition_penalty=1.0, return_history=True, verbose=False
            )
            
            # 判断是否正确：输出中包含正确字符
            is_correct = gt_char in response
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "image": img_path,
                "gt": gt_char,
                "pred": response,
                "correct": is_correct
            })
            
        except Exception as e:
            results.append({
                "image": img_path,
                "gt": gt_char,
                "pred": f"ERROR: {str(e)}",
                "correct": False
            })
            total += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy, correct, total, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, 
                       default='/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset/Test')
    parser.add_argument('--caoshu_ckpt', type=str, 
                       default='/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best_val_1.pt')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_learns', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_samples', type=int, default=None, help='最多测试样本数（默认全部）')
    parser.add_argument('--output', type=str, default='results/caoshu_test_eval.json')
    parser.add_argument('--mode', type=str, choices=['original', 'caoshu'], required=True)
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer, detect_model, generation_config = load_base_model()
    
    if args.mode == 'caoshu':
        old_resampler = model.resampler
        model.resampler = load_caoshu_resampler(args.caoshu_ckpt, args.num_layers, args.num_learns, args.dropout)
        del old_resampler
        gc.collect()
        torch.cuda.empty_cache()
    
    # 评估
    accuracy, correct, total, results = evaluate_model(
        model, tokenizer, detect_model, generation_config,
        args.test_dir, max_samples=args.max_samples
    )
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            "mode": args.mode,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
