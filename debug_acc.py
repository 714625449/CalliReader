"""诊断准确率问题：检查resampler输出是否collapse，以及embedding是否有异常"""
import os, sys, torch, torch.nn.functional as F
PROJECT_ROOT = '/caoshu'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'caoshu'))

from models.model import (
    load_vision_model, load_mlp1, load_perceiver_resampler,
    load_normed_tok_embeddings, load_tokenizer,
)
from config.configu import DOWNSAMPLE_RATIO
from dataset import CaoshuDataset, get_transform
from torch.utils.data import DataLoader

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    new_c = int(c / (scale_factor * scale_factor))
    x = x.reshape(n, new_w, 2, new_h, 2, c)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(n, new_w, new_h, new_c)
    return x

@torch.no_grad()
def get_visual_embed(imgs, vit, mlp1):
    vit_out = vit(imgs).last_hidden_state[:, 1:, :]
    h = w = int(vit_out.shape[1] ** 0.5)
    vit_out = vit_out.view(vit_out.shape[0], h, w, -1)
    vit_out = pixel_shuffle(vit_out, scale_factor=DOWNSAMPLE_RATIO)
    vit_out = vit_out.view(vit_out.shape[0], -1, vit_out.shape[-1])
    vit_out = mlp1(vit_out)
    return vit_out

device = torch.device('cuda')
ckpt_path = '/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best.pt'
data_root = '/root/sj-tmp/datasets/CCC_split'

print("加载模型...")
vit = load_vision_model(location='cuda')
vit.eval()
mlp1 = load_mlp1(downsample_ratio=DOWNSAMPLE_RATIO)
mlp1.eval()

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = ckpt['model_state_dict']
layer_indices = set()
for k in state_dict.keys():
    if k.startswith('layers.'):
        layer_idx = int(k.split('.')[1])
        layer_indices.add(layer_idx)
num_layers = max(layer_indices) + 1
num_learns = state_dict['learns'].shape[0]
print(f"模型: num_layers={num_layers}, num_learns={num_learns}")

resampler = load_perceiver_resampler(path=None, num_layers=num_layers)
if resampler.learns.shape[0] != num_learns:
    resampler.learns = torch.nn.Parameter(torch.randn(num_learns, 4096, device=device, dtype=torch.bfloat16))
resampler = resampler.to(device).to(torch.bfloat16)
resampler.load_state_dict(state_dict)
resampler.eval()

tok_embeddings, _ = load_normed_tok_embeddings(load_checkboard=True, location='cpu')
tok_embeddings = tok_embeddings.to(device).to(torch.bfloat16)
tok_embeddings.eval()
tokenizer = load_tokenizer()

dataset = CaoshuDataset(data_root, 'Validation', transform=get_transform('Validation'))
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

# 取第一个batch
imgs, labels = next(iter(loader))
imgs = imgs.to(device)

with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        vit_feats = get_visual_embed(imgs, vit, mlp1)
        pred = resampler(vit_feats)

pred = pred.float()
print(f"\npred shape: {pred.shape}")  # (B, N, D)

# 检查不同样本的pred是否相同
print("\n===== 样本间pred差异 =====")
for i in range(pred.shape[0]):
    for j in range(i+1, pred.shape[0]):
        diff = (pred[i] - pred[j]).abs().mean().item()
        print(f"  sample {i} vs {j}: mean abs diff = {diff:.6f}")

# 每个样本的N个learns之间的差异
print("\n===== 单个样本内learns差异 =====")
for i in range(min(2, pred.shape[0])):
    p = pred[i]  # (N, D)
    var = p.std(dim=0).mean().item()
    print(f"  sample {i}: learns std over dim=0 = {var:.6f}")

# 预计算embedding，并检查"逄"的embedding
all_chars = [dataset.idx2char[i] for i in range(len(dataset.idx2char))]
all_embeds = []
for i in range(0, len(all_chars), 100):
    batch_chars = all_chars[i:i+100]
    batch_tokens = tokenizer(batch_chars, return_tensors='pt', add_special_tokens=False,
                             padding=True, truncation=True, max_length=4).input_ids[:, 0].to(device)
    with torch.no_grad():
        batch_embeds = tok_embeddings(batch_tokens)
    all_embeds.append(batch_embeds)
all_embeds = torch.cat(all_embeds, dim=0)
all_embeds_norm = F.normalize(all_embeds, dim=-1).float()

# 找"逄"的索引
if '逄' in dataset.idx2char.values():
    pang_idx = [k for k, v in dataset.idx2char.items() if v == '逄'][0]
    print(f"\n===== '逄'的embedding分析 =====")
    print(f"  索引: {pang_idx}")
    pang_embed = all_embeds_norm[pang_idx]
    # 检查它的范数（已归一化，应为1）
    print(f"  范数: {pang_embed.norm().item():.6f}")
    # 与其他embedding的最大相似度
    sims = torch.mm(pang_embed.unsqueeze(0), all_embeds_norm.t())
    print(f"  与所有字符的最大相似度: {sims.max().item():.6f}")
    print(f"  与所有字符的最小相似度: {sims.min().item():.6f}")
    print(f"  与所有字符的相似度均值: {sims.mean().item():.6f}")
    print(f"  相似度>0.9的数量: {(sims[0] > 0.9).sum().item()}")

# 实际推理并看相似度分布
print("\n===== 实际推理相似度分析 =====")
for i in range(min(3, pred.shape[0])):
    p = pred[i].mean(dim=0) if pred.dim() == 3 else pred[i]
    p_norm = F.normalize(p, dim=-1)
    sims = torch.mm(p_norm.unsqueeze(0), all_embeds_norm.t())
    top5_vals, top5_ids = torch.topk(sims[0], k=5)
    true_char = dataset.idx2char[labels[i].item()]
    top5_chars = [dataset.idx2char[int(idx)] for idx in top5_ids]
    print(f"  sample {i}: true='{true_char}'")
    print(f"    top5: {list(zip(top5_chars, top5_vals.cpu().numpy().tolist()))}")
    print(f"    sims stats: max={sims.max().item():.4f}, min={sims.min().item():.4f}, mean={sims.mean().item():.4f}, std={sims.std().item():.4f}")
    print(f"    pred std across queries: {pred[i].float().std(dim=-1).cpu().numpy().tolist()}")

