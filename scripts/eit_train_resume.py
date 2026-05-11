#!/usr/bin/env python3
import os
os.environ['HF_HUB_OFFLINE'] = '1'
"""
Resume e-IT training from checkpoint-600.
Auto-cleans old checkpoints and monitors disk space.
"""
import os
import sys
import json
import shutil
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

sys.path.insert(0, '/caoshu')

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_PATH = '/caoshu/train/samples_eit.json'
MODEL_PATH = '/caoshu/InternVL'
OUTPUT_DIR = '/caoshu/outputs/eit_simple_overfit'
RESUME_CKPT = '/caoshu/outputs/eit_simple_overfit/checkpoint-1700'
BATCH_SIZE = 1
MAX_EPOCHS = 10
LR = 5e-5
SAVE_STEPS = 100
KEEP_CHECKPOINTS = 5          # 最多保留 N 个 checkpoint
MIN_FREE_GB = 5               # 剩余空间低于此值时跳过保存

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Disk utils
# ------------------------------------------------------------------
def get_free_space_gb(path):
    return shutil.disk_usage(path).free / (1024 ** 3)

def cleanup_old_checkpoints(output_dir, keep=KEEP_CHECKPOINTS):
    checkpoints = sorted([
        d for d in os.listdir(output_dir)
        if d.startswith('checkpoint-') and os.path.isdir(os.path.join(output_dir, d))
    ], key=lambda x: int(x.split('-')[1]))
    
    for old_ckpt in checkpoints[:-keep]:
        old_path = os.path.join(output_dir, old_ckpt)
        print(f"[Disk] Removing old checkpoint to save space: {old_path}")
        shutil.rmtree(old_path)

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

# Freeze non-LLM parts
model.vision_model.requires_grad_(False)
model.resampler.requires_grad_(False)
model.mlp1.requires_grad_(False)
model.normed_emb.requires_grad_(False)

# Freeze base LLM
model.language_model.requires_grad_(False)

# ------------------------------------------------------------------
# Resume LoRA from checkpoint-600
# ------------------------------------------------------------------
print(f"Resuming LoRA weights from {RESUME_CKPT}...")
model.language_model = PeftModel.from_pretrained(
    model.language_model,
    RESUME_CKPT,
    is_trainable=True,
)
model.language_model.print_trainable_parameters()

# Enable grad checkpointing
model.language_model.gradient_checkpointing_enable()
model.language_model.enable_input_require_grads()

embeds_token_id = tokenizer.convert_tokens_to_ids('[UNUSED_TOKEN_140]')

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class EITDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        embedding = torch.load(item['embedding'], map_location='cpu')
        gt_text = item['conversations'][1]['value']
        question = item['conversations'][0]['value']
        return {
            'embedding': embedding,
            'gt_text': gt_text,
            'question': question,
        }

def collate_fn(batch):
    return batch

dataset = EITDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ------------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------------
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(
    filter(lambda p: p.requires_grad, model.language_model.parameters()),
    lr=LR
)

model.language_model.train()

# ------------------------------------------------------------------
# Resume state: 539 samples/epoch, 600 steps = 1 full epoch + 61 steps
# We restart from epoch 2 (index 1) and skip first 61 steps.
# ------------------------------------------------------------------
SAMPLES_PER_EPOCH = len(dataset)  # 539
resume_epoch = 1700 // SAMPLES_PER_EPOCH  # epoch index (0-based)
skip_steps = 1700 % SAMPLES_PER_EPOCH

global_step = 1700
print(f"Resuming from epoch {resume_epoch + 1}, skipping first {skip_steps} steps...")

for epoch in range(resume_epoch, MAX_EPOCHS):
    epoch_loss = 0.0
    num_batches = 0
    step_in_epoch = 0
    
    for batch in dataloader:
        step_in_epoch += 1
        
        # Skip already-done steps in the first resumed epoch
        if epoch == resume_epoch and step_in_epoch <= skip_steps:
            continue
        
        item = batch[0]
        embedding = item['embedding'].cuda().to(torch.bfloat16)
        gt_text = item['gt_text']
        question = item['question']
        
        # Build text
        system_text = "<|im_start|>system\n你是书生·浦语，一个有用的人工智能助手。<|im_end|>\n"
        user_text = f"<|im_start|>user\n{question}<|im_end|>\n"
        assistant_text = f"<|im_start|>assistant\n{gt_text}<|im_end|>"
        full_text = system_text + user_text + assistant_text
        
        tokens = tokenizer(full_text, return_tensors='pt', add_special_tokens=False)
        input_ids = tokens['input_ids'].cuda()
        
        # Labels: mask prefix
        prefix_text = system_text + user_text
        prefix_tokens = tokenizer(prefix_text, return_tensors='pt', add_special_tokens=False)
        prefix_len = prefix_tokens['input_ids'].shape[1]
        labels = input_ids.clone()
        labels[:, :prefix_len] = -100
        
        # Embeddings
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
        
        # Forward
        outputs = model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=tokens['attention_mask'].cuda(),
            labels=labels,
            use_cache=False,
        )
        loss = outputs.loss
        
        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        global_step += 1
        epoch_loss += loss.item()
        num_batches += 1
        
        if global_step % 10 == 0:
            print(f"Epoch {epoch + 1}/{MAX_EPOCHS} Step {global_step} | Loss: {loss.item():.4f}")
        
        if global_step % SAVE_STEPS == 0:
            free_gb = get_free_space_gb(OUTPUT_DIR)
            if free_gb < MIN_FREE_GB:
                print(f"[Disk] WARNING: Only {free_gb:.2f}GB free (< {MIN_FREE_GB}GB), skipping checkpoint save")
            else:
                save_path = os.path.join(OUTPUT_DIR, f'checkpoint-{global_step}')
                model.language_model.save_pretrained(save_path)
                cleanup_old_checkpoints(OUTPUT_DIR, keep=KEEP_CHECKPOINTS)
                print(f"[Disk] {free_gb:.2f}GB free | Saved checkpoint to {save_path}")
    
    avg_loss = epoch_loss / max(num_batches, 1)
    print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f}")
    # Reset skip after first resumed epoch
    skip_steps = 0

# Final save
free_gb = get_free_space_gb(OUTPUT_DIR)
if free_gb >= MIN_FREE_GB:
    save_path = os.path.join(OUTPUT_DIR, 'final')
    model.language_model.save_pretrained(save_path)
    cleanup_old_checkpoints(OUTPUT_DIR, keep=KEEP_CHECKPOINTS)
    print(f"Training complete. Final model saved to {save_path}")
else:
    print(f"[Disk] WARNING: Only {free_gb:.2f}GB free, skipping final save")
