"""
安全测试草书模型：不修改原模型文件，运行时动态替换
"""
import os
import sys
import torch
import json
from pathlib import Path

# 强制使用本地路径，避免污染原模型
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, '/workspace/CalliReader')

def test_with_caoshu():
    """加载原模型，但替换 resampler 为草书版本"""
    
    print("=" * 60)
    print("加载原 CalliReader + 草书 CalliAlign")
    print("=" * 60)
    
    # 1. 加载原模型（标准流程）
    from transformers import AutoModel, AutoTokenizer
    from ultralytics import YOLO
    from config.configu import INTERNVL_PATH, YOLO_CHECKPOINT, SEED
    from inference import setup_logger, set_seed, is_image, get_image_paths
    
    set_seed(SEED)
    
    print(f"基础模型: {INTERNVL_PATH}")
    model = AutoModel.from_pretrained(
        INTERNVL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True
    ).eval().cuda()
    
    # 2. 关键：替换 resampler 权重（内存中修改，不保存）
    caoshu_ckpt = '/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best.pt'
    print(f"加载草书权重: {caoshu_ckpt}")
    
    ckpt = torch.load(caoshu_ckpt, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    
    # 找到并替换 resampler
    replaced = False
    for name, module in model.named_modules():
        if name == 'resampler' and hasattr(module, 'load_state_dict'):
            module.load_state_dict(state_dict, strict=True)
            print(f"✓ 成功替换: {name}")
            replaced = True
            break
    
    if not replaced:
        print("✗ 未找到 resampler 模块")
        return None
    
    # 3. 准备其他组件
    tokenizer = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True)
    detect_model = YOLO(str(YOLO_CHECKPOINT))
    
    generation_config = dict(
        num_beams=1,
        max_new_tokens=1024,
        do_sample=False,
    )
    
    return model, tokenizer, detect_model, generation_config

def batch_test(examples_dir='./examples'):
    """批量测试"""
    
    # 加载模型（带草书权重）
    result = test_with_caoshu()
    if result is None:
        return
    model, tokenizer, detect_model, generation_config = result
    
    # 获取所有测试图
    image_paths = sorted(list(Path(examples_dir).glob('*.jpg')) + 
                        list(Path(examples_dir).glob('*.png')))
    
    print(f"\n找到 {len(image_paths)} 张测试图")
    print("开始识别...\n")
    
    results = []
    
    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {img_path.name}")
        print("-" * 60)
        
        try:
            # 直接调用 model.chat_ocr（不通过 inference.py，避免重新加载模型）
            response, history = model.chat_ocr(
                tokenizer,
                detect_model,
                str(img_path),
                "这幅书法作品中的文字是什么？",
                generation_config,
                use_p=True,
                hard_vq=False,
                drop_zero=False,
                repetition_penalty=1.0,
                return_history=True,
                verbose=False
            )
            
            print(f"识别结果: {response}")
            
            results.append({
                "image": img_path.name,
                "response": response
            })
            
        except Exception as e:
            print(f"错误: {e}")
            results.append({
                "image": img_path.name,
                "response": f"ERROR: {str(e)}"
            })
        
        print()
    
    # 保存结果（不覆盖原模型结果）
    output_file = 'results/test_caoshu_isolated.json'
    os.makedirs('results', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("=" * 60)
    print(f"结果保存: {output_file}")
    print("提示: 对比文件")
    print("  - results/test_original_Original.json (原模型)")
    print("  - results/test_caoshu_isolated.json (草书模型)")
    print("=" * 60)
    
    return results

if __name__ == '__main__':
    batch_test('./examples')
