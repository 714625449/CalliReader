import sys
sys.path.insert(0, '/workspace/CalliReader')
sys.path.insert(0, '/workspace/CalliReader/caoshu')

import torch
import torch.nn.functional as F

print("=== 测试两种 Resampler 加载方式 ===\n")

# 方式1：pipeline.py 的方式（单独创建）
print("方式1: load_perceiver_resampler(None) + load_state_dict")
from models.model import load_perceiver_resampler

resampler1 = load_perceiver_resampler(None, num_layers=4)
print(f"  结构类型: {type(resampler1)}")
print(f"  参数字典键数: {len(list(resampler1.state_dict().keys()))}")
print(f"  示例键: {list(resampler1.state_dict().keys())[:3]}")

# 加载 checkpoint 看是否匹配
ckpt = torch.load('/root/sj-tmp/checkpoints/CalliReader/params/callialign.pth', map_location='cpu')
if 'model_state_dict' in ckpt:
    ckpt_keys = list(ckpt['model_state_dict'].keys())
    print(f"  Checkpoint 键数: {len(ckpt_keys)}")
    print(f"  Checkpoint 示例键: {ckpt_keys[:3]}")
    
    # 对比键名
    model_keys = set(resampler1.state_dict().keys())
    ckpt_keys_clean = set([k.replace('module.', '') for k in ckpt_keys])
    
    common = model_keys & ckpt_keys_clean
    only_model = model_keys - ckpt_keys_clean
    only_ckpt = ckpt_keys_clean - model_keys
    
    print(f"\n  匹配键数: {len(common)}")
    print(f"  只在模型中: {len(only_model)}")
    print(f"  只在ckpt中: {len(only_ckpt)}")
    
    if only_model:
        print(f"    示例: {list(only_model)[:3]}")
    if only_ckpt:
        print(f"    示例: {list(only_ckpt)[:3]}")

print("\n=== 方式2: 参考 test_caoshu_safe.py ===")
print("test_caoshu_safe.py 是加载完整 InternVL，然后替换内部 resampler")

# 查看 InternVL 内部的 resampler 结构
print("\n需要加载 InternVL 查看内部 resampler...")

# 模拟 test_caoshu_safe.py 的方式
from transformers import AutoModel, AutoConfig

try:
    # 尝试加载 InternVL 的 vision config
    from models.model import TOKENIZER_PATH
    print(f"TOKENIZER_PATH: {TOKENIZER_PATH}")
    
    # 如果 test_caoshu_safe.py 加载了完整模型，我们需要看那个模型的 resampler
    print("\n请提供 test_caoshu_safe.py 中加载 InternVL 的代码片段")
    print("或者运行: sed -n '1,80p' test_caoshu_safe.py")
    
except Exception as e:
    print(f"查看失败: {e}")

print("\n=== 结论 ===")
print("如果方式1显示 '匹配键数' 很少，说明结构不匹配")
print("如果显示 '匹配键数' 很多但结果仍错，说明权重值有问题")
