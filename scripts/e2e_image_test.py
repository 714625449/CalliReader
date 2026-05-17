#!/usr/bin/env python3
"""
端到端图片识别测试：直接用 InternVL + LoRA 对草书图片做 OCR
"""
import os
import sys
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, '/caoshu')

# ------------------------------------------------------------------
# Image preprocessing (from InternVL README)
# ------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_PATH = '/caoshu/InternVL'
LORA_PATH = '/caoshu/outputs/eit_simple_overfit/final'
TEST_IMAGES = [
    '/caoshu/imgs/2.jpg',
    '/caoshu/imgs/6.jpg',
    '/caoshu/imgs/10.jpg',
]
QUESTION = '读一下图片中书写的文字。'

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
print("Loading InternVL base model...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# ------------------------------------------------------------------
# Baseline inference (no LoRA)
# ------------------------------------------------------------------
print("\n" + "="*60)
print("BASELINE (InternVL without LoRA)")
print("="*60)

def run_inference(model, tokenizer, image_path):
    pixel_values = load_image(image_path, max_num=6).cuda().to(torch.bfloat16)
    generation_config = dict(
        max_new_tokens=256,
        do_sample=False,
        num_beams=1,
    )
    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=QUESTION,
            generation_config=generation_config,
            verbose=False,
        )
    return response

for img_path in TEST_IMAGES:
    if not Path(img_path).exists():
        print(f"Skip missing: {img_path}")
        continue
    print(f"\n[{Path(img_path).name}]")
    try:
        response = run_inference(model, tokenizer, img_path)
        print(f"  Response: {response}")
    except Exception as e:
        print(f"  Error: {e}")

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
print("INTERNVL + e-IT LoRA")
print("="*60)

for img_path in TEST_IMAGES:
    if not Path(img_path).exists():
        continue
    print(f"\n[{Path(img_path).name}]")
    try:
        response = run_inference(model, tokenizer, img_path)
        print(f"  Response: {response}")
    except Exception as e:
        print(f"  Error: {e}")

print("\nDone.")
