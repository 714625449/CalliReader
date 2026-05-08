"""
对比测试：原 CalliReader vs 草书训练后的 CalliReader
修复：正确加载 num_layers=8, num_learns=12 的 Caoshu Resampler
"""
import os
import sys
import torch
import json
import gc
from pathlib import Path
from PIL import Image

# 添加路径
PROJECT_ROOT = '/workspace/CalliReader'
sys.path.insert(0, PROJECT_ROOT)

from inference import setup_logger, set_seed, is_image, get_image_paths
from config.configu import SEED
from models.model import load_perceiver_resampler
import argparse

# 设置
set_seed(SEED)
os.makedirs('results', exist_ok=True)

def load_base_model():
    """加载基础 InternVL 模型（不含自定义 resampler 权重）"""
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
    
    generation_config = dict(
        num_beams=1,
        max_new_tokens=1024,
        do_sample=False,
    )
    
    return model, tokenizer, detect_model, generation_config

def load_caoshu_resampler(caoshu_ckpt_path, num_layers=8, num_learns=12, dropout=0.1):
    """加载训练好的 Caoshu Resampler"""
    print(f"Loading Caoshu CalliAlign from {caoshu_ckpt_path}")
    caoshu_ckpt = torch.load(caoshu_ckpt_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in caoshu_ckpt:
        state_dict = caoshu_ckpt['model_state_dict']
    else:
        state_dict = caoshu_ckpt
    
    # 创建正确尺寸的 resampler
    resampler = load_perceiver_resampler(
        path=None,
        num_layers=num_layers,
        num_learns=num_learns,
        dropout=dropout,
        checkpoint=None
    )
    
    # 加载权重
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    resampler.load_state_dict(state_dict, strict=True)
    resampler = resampler.to(torch.bfloat16).cuda()
    
    print(f"Successfully loaded Caoshu CalliAlign (layers={num_layers}, learns={num_learns})")
    return resampler

def test_single_image(image_path, model, tokenizer, detect_model, generation_config, prompt="这幅书法作品中的文字是什么？", use_p=True):
    """测试单张图片"""
    try:
        response, history = model.chat_ocr(
            tokenizer, 
            detect_model,
            image_path, 
            prompt, 
            generation_config,
            use_p=use_p,
            hard_vq=False,
            drop_zero=False,
            repetition_penalty=1.0,
            return_history=True,
            verbose=False
        )
        return response
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return f"ERROR: {str(e)}"

def batch_test(model, tokenizer, detect_model, generation_config, examples_dir, output_name="comparison", model_type="Original"):
    """批量测试"""
    image_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_paths.extend(Path(examples_dir).glob(ext))
    
    image_paths = sorted([str(p) for p in image_paths])
    print(f"Found {len(image_paths)} images in {examples_dir}")
    
    print(f"\n{'='*60}")
    print(f"Testing with: {model_type} CalliReader")
    print(f"{'='*60}")
    
    results = []
    for img_path in image_paths:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        response = test_single_image(img_path, model, tokenizer, detect_model, generation_config)
        print(f"Result: {response[:100]}...")
        
        results.append({
            "image": os.path.basename(img_path),
            "response": response
        })
    
    output_file = f"results/{output_name}_{model_type}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['original', 'caoshu', 'both'], default='both')
    parser.add_argument('--examples_dir', type=str, default='./examples')
    parser.add_argument('--caoshu_ckpt', type=str, 
                       default='/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best.pt')
    parser.add_argument('--num_layers', type=int, default=8, help='Resampler层数')
    parser.add_argument('--num_learns', type=int, default=12, help='Query数量')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    args = parser.parse_args()
    
    # 加载基础模型（只加载一次）
    model, tokenizer, detect_model, generation_config = load_base_model()
    
    if args.mode in ['original', 'both']:
        print("\n" + "="*60)
        print("STEP 1: 测试原 CalliReader（混合字体训练）")
        print("="*60)
        original_results = batch_test(
            model, tokenizer, detect_model, generation_config,
            args.examples_dir, "test_original", "Original"
        )
    
    if args.mode in ['caoshu', 'both']:
        print("\n" + "="*60)
        print("STEP 2: 测试草书 CalliReader（纯草书训练）")
        print("="*60)
        
        # 替换 resampler
        old_resampler = model.resampler
        model.resampler = load_caoshu_resampler(
            args.caoshu_ckpt,
            num_layers=args.num_layers,
            num_learns=args.num_learns,
            dropout=args.dropout
        )
        
        # 尝试释放旧的 resampler 显存
        del old_resampler
        gc.collect()
        torch.cuda.empty_cache()
        
        caoshu_results = batch_test(
            model, tokenizer, detect_model, generation_config,
            args.examples_dir, "test_caoshu", "Caoshu"
        )
    
    if args.mode == 'both':
        print("\n" + "="*60)
        print("对比完成！请查看：")
        print("  - results/test_original_Original.json")
        print("  - results/test_caoshu_Caoshu.json")
        print("="*60)

if __name__ == '__main__':
    main()
