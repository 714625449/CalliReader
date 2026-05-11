#!/usr/bin/env python3
"""
Lightweight e-IT training script without xtuner.
Directly uses PyTorch + PEFT to fine-tune LLM with precomputed embeddings.
"""
import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, '/caoshu')

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_PATH = '/caoshu/train/samples_eit.json'
MODEL_PATH = '/caoshu/InternVL'
OUTPUT_DIR = '/caoshu/outputs/eit_simple_overfit'
BATCH_SIZE = 1
MAX_EPOCHS = 10
LR = 5e-5
LORA_R = 128
LORA_ALPHA = 256
SAVE_STEPS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
print("Loading model...")
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

# Freeze base LLM, then apply LoRA
model.language_model.requires_grad_(False)
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["wqkv", "wo", "w1", "w2", "w3"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model.language_model = get_peft_model(model.language_model, peft_config)
model.language_model.print_trainable_parameters()

# Enable grad checkpointing to save memory
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
        embedding = torch.load(item['embedding'], map_location='cpu')  # (N*3, 4096)
        gt_text = item['conversations'][1]['value']
        question = item['conversations'][0]['value']
        return {
            'embedding': embedding,
            'gt_text': gt_text,
            'question': question,
        }

def collate_fn(batch):
    return batch  # list of dicts

dataset = EITDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
# Use 8-bit AdamW to save optimizer state memory (~50% reduction)
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(
    filter(lambda p: p.requires_grad, model.language_model.parameters()),
    lr=LR
)

model.language_model.train()
global_step = 0

for epoch in range(MAX_EPOCHS):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        item = batch[0]  # batch_size=1
        embedding = item['embedding'].cuda().to(torch.bfloat16)  # (N*3, 4096)
        gt_text = item['gt_text']
        question = item['question']
        
        # Build full text: system + user + assistant
        system_text = "<|im_start|>system\n你是书生·浦语，一个有用的人工智能助手。<|im_end|>\n"
        user_text = f"<|im_start|>user\n{question}<|im_end|>\n"
        assistant_text = f"<|im_start|>assistant\n{gt_text}<|im_end|>"
        
        full_text = system_text + user_text + assistant_text
        
        # Tokenize
        tokens = tokenizer(full_text, return_tensors='pt', add_special_tokens=False)
        input_ids = tokens['input_ids'].cuda()  # (1, seq_len)
        
        # Build labels: mask out system + user, only train on assistant
        prefix_text = system_text + user_text
        prefix_tokens = tokenizer(prefix_text, return_tensors='pt', add_special_tokens=False)
        prefix_len = prefix_tokens['input_ids'].shape[1]
        
        labels = input_ids.clone()
        labels[:, :prefix_len] = -100
        
        # Get input embeddings (embedding layer is frozen, no grad needed)
        input_embeds = model.language_model.get_input_embeddings()(input_ids)  # (1, seq_len, 4096)
        
        # Inject precomputed embedding at [UNUSED_TOKEN_140] positions
        input_embeds = input_embeds.clone()
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        flat_ids = input_ids.reshape(B * N)
        
        selected = (flat_ids == embeds_token_id)
        num_selected = selected.sum().item()
        num_embed = embedding.shape[0]
        
        if num_selected != num_embed:
            print(f"WARNING: selected={num_selected}, embed={num_embed}, gt_len={len(gt_text)}")
            # Truncate to min
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
            print(f"Epoch {epoch+1}/{MAX_EPOCHS} Step {global_step} | Loss: {loss.item():.4f}")
        
        if global_step % SAVE_STEPS == 0:
            save_path = os.path.join(OUTPUT_DIR, f'checkpoint-{global_step}')
            model.language_model.save_pretrained(save_path)
            print(f"Saved checkpoint to {save_path}")
    
    avg_loss = epoch_loss / max(num_batches, 1)
    print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

# Final save
save_path = os.path.join(OUTPUT_DIR, 'final')
model.language_model.save_pretrained(save_path)
print(f"Training complete. Final model saved to {save_path}")
