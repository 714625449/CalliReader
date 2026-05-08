import sys
sys.path.insert(0, '/workspace/CalliReader')
sys.path.insert(0, '/workspace/CalliReader/caoshu')

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

print("=== 终极对比测试 ===\n")

device = torch.device('cuda')

# 加载字图（pipeline 裁剪的）
img_path = 'outputs/test_v2.2/chars/char_001.jpg'
img = Image.open(img_path).convert('RGB')
print(f"测试图像: {img_path}, 尺寸: {img.size}")

# ========== 方式A：pipeline 的完整流程 ==========
print("\n--- 方式A: Pipeline 方式 ---")
from caoshu.pipeline import CalliReaderPipeline

# 只初始化，不处理完整图
pipeline = CalliReaderPipeline(
    yolo_model_path='params/best.pt',
    checkpoint_path='/root/sj-tmp/checkpoints/CalliReader/params/callialign.pth',
    data_root='/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset',
    conf_thres=0.25
)

# 使用 pipeline 的 transform 和 recognize_single_char
from torchvision import transforms as T
transform_A = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tensor_A = transform_A(img).unsqueeze(0).to(device).to(torch.bfloat16)

with torch.no_grad():
    from caoshu.pipeline import get_visual_embed
    vit_out_A = get_visual_embed(tensor_A, pipeline.vit, pipeline.mlp1)
    pred_A = pipeline.resampler(vit_out_A)
    if pred_A.dim() == 3:
        pred_A = pred_A.mean(dim=1)
    pred_norm_A = F.normalize(pred_A, dim=-1)
    sim_A = torch.mm(pred_norm_A, pipeline.all_char_embeds_norm.t())
    top3_A = torch.topk(sim_A[0], k=3)
    probs_A = F.softmax(top3_A.values, dim=-1)
    
    print(f"VIT输出形状: {vit_out_A.shape}")
    print(f"VIT输出范围: [{vit_out_A.min():.3f}, {vit_out_A.max():.3f}]")
    print(f"相似度范围: [{sim_A.min():.3f}, {sim_A.max():.3f}]")
    print(f"Top3: ", end="")
    for idx, prob in zip(top3_A.indices, probs_A):
        char = pipeline.idx2char[idx.item()]
        print(f"{char}({prob*100:.0f}%) ", end="")
    print()

# ========== 方式B：test_caoshu_safe.py 的方式 ==========
print("\n--- 方式B: test_caoshu_safe.py 方式 ---")

# 查看 test_caoshu_safe.py 的加载方式（从前面grep知道它替换的是InternVL内部的resampler）
# 我们需要模拟它的方式

# 首先，test_caoshu_safe.py 可能用了不同的transform
from caoshu.dataset import get_transform
transform_B = get_transform('Validation')

tensor_B = transform_B(img).unsqueeze(0).to(device).to(torch.bfloat16)

print(f"Transform A 输出: shape={tensor_A.shape}, range=[{tensor_A.min():.3f}, {tensor_A.max():.3f}]")
print(f"Transform B 输出: shape={tensor_B.shape}, range=[{tensor_B.min():.3f}, {tensor_B.max():.3f}]")

# 差异
if tensor_A.shape == tensor_B.shape:
    diff = (tensor_A - tensor_B).abs().max()
    print(f"Tensor 最大差异: {diff:.6f}")
    if diff < 0.001:
        print("✓ Transform 结果几乎相同")
    else:
        print("✗ Transform 结果不同！")

# ========== 方式C：直接用 test_caoshu_safe.py 的模型 ==========
print("\n--- 方式C: 运行 test_caoshu_safe.py 评估单张图 ---")
print("请执行: python test_caoshu_safe.py --ckpt /root/sj-tmp/checkpoints/CalliReader/params/callialign.pth --num_test 1")
print("并观察它处理单张图时的 Top-3 结果")

print("\n=== 分析 ===")
print("如果方式A和方式B的Top3都是'一'，但方式C正确，说明问题在模型加载")
print("如果方式A的VIT输出范围异常（全0或很大），说明VIT权重没加载")
