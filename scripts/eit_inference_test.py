#!/usr/bin/env python3
"""
推理测试：对比基线 InternVL 与 e-IT LoRA 微调后的模型
"""
import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, '/caoshu')

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_PATH = '/caoshu/train/samples_eit.json'
MODEL_PATH = '/caoshu/InternVL'
LORA_PATH = '/caoshu/outputs/eit_simple_overfit/final'
NUM_SAMPLES = 5  # 测试样本数

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
print("Loading base model...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
embeds_token_id = tokenizer.convert_tokens_to_ids('[UNUSED_TOKEN_140]')

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 选前 NUM_SAMPLES 个样本（可改成随机）
test_samples = data[:NUM_SAMPLES]

# ------------------------------------------------------------------
# Inference helper
# ------------------------------------------------------------------
def infer(model, item, use_lora=False):
    """对单个样本做推理"""
    embedding = torch.load(item['embedding'], map_location='cpu').cuda().to(torch.bfloat16)
    question = item['conversations'][0]['value']
    gt_text = item['conversations'][1]['value']
    
    # Build prompt (same as training)
    system_text = "<|im_start|>system\n你是书生·浦语，一个有用的人工智能助手。<|im_end|>\n"
    user_text = f"<|im_start|>user\n{question}<|im_end|>\n"
    assistant_prefix = "<|im_start|>assistant\n"
    prompt_text = system_text + user_text + assistant_prefix
    
    tokens = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False)
    input_ids = tokens['input_ids'].cuda()
    
    # Get embeddings and inject
    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    input_embeds = input_embeds.clone()
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)
    flat_ids = input_ids.reshape(B * N)
    
    selected = (flat_ids == embeds_token_id)
    num_selected = selected.sum().item()
    num_embed = embedding.shape[0]
    
    if num_selected != num_embed:
        n = min(num_selected, num_embed)
        selected_idx = torch.where(selected)[0][:n]
        input_embeds[selected_idx] = embedding[:n]
    else:
        input_embeds[selected] = embedding
    
    input_embeds = input_embeds.reshape(B, N, C)
    
    # Generate
    with torch.no_grad():
        outputs = model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=tokens['attention_mask'].cuda(),
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0, input_ids.shape[1]:]
    pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return {
        'question': question,
        'gt': gt_text,
        'pred': pred_text.strip(),
        'match': pred_text.strip() == gt_text.strip()
    }

# ------------------------------------------------------------------
# Baseline inference
# ------------------------------------------------------------------
print("\n" + "="*60)
print("BASELINE (no LoRA)")
print("="*60)
baseline_results = []
for i, item in enumerate(test_samples):
    print(f"\n[Sample {i+1}/{NUM_SAMPLES}]")
    result = infer(model, item, use_lora=False)
    baseline_results.append(result)
    print(f"Question: {result['question'][:50]}...")
    print(f"GT:       {result['gt']}")
    print(f"Pred:     {result['pred']}")
    print(f"Match:    {'✅' if result['match'] else '❌'}")

# ------------------------------------------------------------------
# Load LoRA
# ------------------------------------------------------------------
print("\n" + "="*60)
print("Loading LoRA weights...")
print("="*60)
model.language_model = PeftModel.from_pretrained(
    model.language_model,
    LORA_PATH,
)
model.language_model = model.language_model.cuda().eval()
print("LoRA loaded.")

# ------------------------------------------------------------------
# LoRA inference
# ------------------------------------------------------------------
print("\n" + "="*60)
print("e-IT LoRA FINE-TUNED")
print("="*60)
lora_results = []
for i, item in enumerate(test_samples):
    print(f"\n[Sample {i+1}/{NUM_SAMPLES}]")
    result = infer(model, item, use_lora=True)
    lora_results.append(result)
    print(f"Question: {result['question'][:50]}...")
    print(f"GT:       {result['gt']}")
    print(f"Pred:     {result['pred']}")
    print(f"Match:    {'✅' if result['match'] else '❌'}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
baseline_acc = sum(1 for r in baseline_results if r['match']) / len(baseline_results)
lora_acc = sum(1 for r in lora_results if r['match']) / len(lora_results)
print(f"Baseline accuracy: {baseline_acc:.1%} ({sum(1 for r in baseline_results if r['match'])}/{len(baseline_results)})")
print(f"LoRA accuracy:     {lora_acc:.1%} ({sum(1 for r in lora_results if r['match'])}/{len(lora_results)})")

# Show all comparisons
print("\nDetailed comparison:")
for i in range(len(test_samples)):
    b = baseline_results[i]
    l = lora_results[i]
    print(f"\n  Sample {i+1}: GT = '{b['gt']}'")
    print(f"    Baseline: '{b['pred']}' {'✅' if b['match'] else '❌'}")
    print(f"    LoRA:     '{l['pred']}' {'✅' if l['match'] else '❌'}")
