"""
对比测试：原 CalliReader vs 草书训练后的 CalliReader
"""
import os
import sys
import torch
import json
from pathlib import Path
from PIL import Image

# 添加路径
PROJECT_ROOT = '/workspace/CalliReader'
sys.path.insert(0, PROJECT_ROOT)

# 导入原 inference 的函数（需要修改）
from inference import setup_logger, set_seed, is_image, get_image_paths
from config.configu import SEED
import argparse

# 设置
set_seed(SEED)
os.makedirs('results', exist_ok=True)

def load_model_with_caoshu_ckpt(caoshu_ckpt_path=None):
    """
    加载模型，如果提供了 caoshu_ckpt_path，则替换 CalliAlign
    """
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
    
    # 关键：替换 CalliAlign 权重
    if caoshu_ckpt_path and os.path.exists(caoshu_ckpt_path):
        print(f"Loading Caoshu CalliAlign from {caoshu_ckpt_path}")
        caoshu_ckpt = torch.load(caoshu_ckpt_path, map_location='cpu', weights_only=False)
        
        # 获取 perceiver_resampler 的状态
        if 'model_state_dict' in caoshu_ckpt:
            state_dict = caoshu_ckpt['model_state_dict']
        else:
            state_dict = caoshu_ckpt
            
        # 替换到模型的 vision_model 中的 perceiver_resampler
        # 需要找到正确的路径
        try:
            # 尝试直接替换 resampler
            model.resampler.load_state_dict(state_dict, strict=False)
            print("Successfully loaded Caoshu CalliAlign")
        except Exception as e:
            print(f"Warning: Could not load Caoshu weights via model.resampler: {e}")
            # 尝试 vision_model 路径
            try:
                if hasattr(model, 'vision_model') and hasattr(model.vision_model, 'resampler'):
                    model.vision_model.resampler.load_state_dict(state_dict, strict=False)
                    print("Successfully loaded Caoshu CalliAlign via vision_model.resampler")
                else:
                    print("Using original CalliAlign")
            except Exception as e2:
                print(f"Warning: Could not load Caoshu weights: {e2}")
                print("Using original CalliAlign")
    
    detect_model = YOLO(str(YOLO_CHECKPOINT))
    
    generation_config = dict(
        num_beams=1,
        max_new_tokens=1024,
        do_sample=False,
    )
    
    return model, tokenizer, detect_model, generation_config

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

def batch_test(examples_dir, caoshu_ckpt_path=None, output_name="comparison"):
    """
    批量测试
    """
    # 获取所有图片
    image_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_paths.extend(Path(examples_dir).glob(ext))
    
    image_paths = sorted([str(p) for p in image_paths])
    print(f"Found {len(image_paths)} images in {examples_dir}")
    
    # 加载模型
    model_type = "Caoshu" if caoshu_ckpt_path else "Original"
    print(f"\n{'='*60}")
    print(f"Testing with: {model_type} CalliReader")
    print(f"{'='*60}")
    
    model, tokenizer, detect_model, generation_config = load_model_with_caoshu_ckpt(caoshu_ckpt_path)
    
    # 测试所有图片
    results = []
    for img_path in image_paths:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        response = test_single_image(img_path, model, tokenizer, detect_model, generation_config)
        print(f"Result: {response[:100]}...")  # 只打印前100字符
        
        results.append({
            "image": os.path.basename(img_path),
            "response": response
        })
    
    # 保存结果
    output_file = f"results/{output_name}_{model_type}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['original', 'caoshu', 'both'], default='both',
                       help='original: 只测原模型, caoshu: 只测草书模型, both: 都测')
    parser.add_argument('--examples_dir', type=str, default='./examples',
                       help='测试图片目录')
    parser.add_argument('--caoshu_ckpt', type=str, 
                       default='/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best.pt',
                       help='草书模型 checkpoint 路径')
    args = parser.parse_args()
    
    if args.mode in ['original', 'both']:
        print("\n" + "="*60)
        print("STEP 1: 测试原 CalliReader（混合字体训练）")
        print("="*60)
        original_results = batch_test(args.examples_dir, None, "test_original")
    
    if args.mode in ['caoshu', 'both']:
        print("\n" + "="*60)
        print("STEP 2: 测试草书 CalliReader（纯草书训练）")
        print("="*60)
        caoshu_results = batch_test(args.examples_dir, args.caoshu_ckpt, "test_caoshu")
    
    if args.mode == 'both':
        print("\n" + "="*60)
        print("对比完成！请查看：")
        print("  - results/test_original_Original.json")
        print("  - results/test_caoshu_Caoshu.json")
        print("="*60)
