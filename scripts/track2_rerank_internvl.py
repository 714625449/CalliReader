#!/usr/bin/env python3
"""Track 2: 用 InternVL (InternLM2.5-7B) 做纯文本 rerank"""

import sys
import json
import torch

sys.path.insert(0, '/caoshu')

from transformers import AutoTokenizer, AutoModel

print("[Rerank] Loading InternVL...")
model_path = '/root/sj-tmp/callireader_models/InternVL'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True
).eval()

print("[Rerank] Model loaded.")
print(f"[Rerank] GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

# 读取 Track 2 原始结果
with open('/root/sj-tmp/p2b_raw_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 触发阈值：Top-1 confidence < 0.3 才 rerank
RERANK_THRESHOLD = 0.30

def build_prompt(prev_chars, candidates):
    """构建 rerank prompt"""
    cand_str = '\n'.join([f"{i+1}. {c['char']}" for i, c in enumerate(candidates)])
    prompt = f"""这是一幅书法作品的 OCR 识别结果。以下识别可能有错，请谨慎判断。

已识别的前文（可能包含错误）：{prev_chars}

当前位置的字，视觉模型给出的候选（按置信度从高到低）：
{cand_str}

请根据上下文语义和常见书法内容（古诗、经典文本），判断当前位置最合理的字是哪一个。只输出字本身，不要解释。

答案："""
    return prompt

def rerank_char(prev_chars, candidates):
    """调用 InternVL 做 rerank，返回选中的字"""
    prompt = build_prompt(prev_chars, candidates)
    
    # InternVL 纯文本生成
    response = model.chat(
        tokenizer,
        None,  # pixel_values=None 表示纯文本
        prompt,
        generation_config=dict(max_new_tokens=10, do_sample=False)
    )
    
    # 清理输出：只取第一个非空字符
    response = response.strip().replace('\n', '').replace(' ', '')
    if len(response) > 0:
        return response[0]
    return candidates[0]['char']  # fallback

# 处理每张图
all_results = []
total_reranked = 0
total_low_conf = 0

for img_data in data:
    chars = img_data['chars']
    reranked_chars = []
    
    for i, char_info in enumerate(chars):
        candidates = char_info['top_candidates']
        best_conf = candidates[0]['confidence']
        
        # 构建前文（前 5 个字）
        prev_start = max(0, i - 5)
        prev_chars = ''.join([chars[j]['best_char'] for j in range(prev_start, i)])
        
        if best_conf < RERANK_THRESHOLD:
            total_low_conf += 1
            try:
                new_char = rerank_char(prev_chars, candidates)
                # 确认 new_char 在 candidates 中
                cand_chars = [c['char'] for c in candidates]
                if new_char in cand_chars:
                    char_info['reranked_char'] = new_char
                    char_info['reranked_from'] = candidates[0]['char']
                    total_reranked += 1
                else:
                    char_info['reranked_char'] = candidates[0]['char']
                    char_info['reranked_from'] = None
            except Exception as e:
                print(f"  [ERR] rerank failed: {e}")
                char_info['reranked_char'] = candidates[0]['char']
                char_info['reranked_from'] = None
        else:
            char_info['reranked_char'] = candidates[0]['char']
            char_info['reranked_from'] = None
        
        reranked_chars.append(char_info)
    
    all_results.append({
        'image': img_data['image'],
        'total_chars': img_data['total_chars'],
        'chars': reranked_chars
    })

# 保存结果
out_path = '/root/sj-tmp/p2b_reranked_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n[Rerank] Done!")
print(f"[Rerank] Total chars: {sum(r['total_chars'] for r in data)}")
print(f"[Rerank] Low confidence (< {RERANK_THRESHOLD}): {total_low_conf}")
print(f"[Rerank] Actually reranked: {total_reranked}")
print(f"[Rerank] Saved to {out_path}")
