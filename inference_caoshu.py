#!/usr/bin/env python3
"""
草书 CalliReader 推理脚本
支持加载训练好的 Caoshu CalliAlign 权重进行整篇作品识别
"""
import os
import sys
import argparse
import torch
import gc

PROJECT_ROOT = '/workspace/CalliReader'
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModel, AutoTokenizer
from ultralytics import YOLO
from pathlib import Path
from config.configu import INTERNVL_PATH, YOLO_CHECKPOINT, SEED
from models.model import load_perceiver_resampler
from inference import single_rec, folder_rec, set_seed

set_seed(SEED)


def load_model_with_caoshu_ckpt(caoshu_ckpt_path=None, num_layers=8, num_learns=12, dropout=0.1):
    """
    加载 InternVL 模型，可选替换为草书训练后的 CalliAlign
    """
    print(f"Loading base model from {INTERNVL_PATH}...")
    model = AutoModel.from_pretrained(
        INTERNVL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True)

    # 替换 resampler
    if caoshu_ckpt_path and os.path.exists(caoshu_ckpt_path):
        print(f"Loading Caoshu CalliAlign from {caoshu_ckpt_path}")
        ckpt = torch.load(caoshu_ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # 创建新 resampler
        old_resampler = model.resampler
        model.resampler = load_perceiver_resampler(
            path=None,
            num_layers=num_layers,
            num_learns=num_learns,
            dropout=dropout,
            checkpoint=None
        )
        model.resampler.load_state_dict(state_dict, strict=True)
        model.resampler = model.resampler.to(torch.bfloat16).cuda()

        # 注意：不修改 num_image_token，保持为 3
        # num_image_token 控制的是 VIT 特征的分块数，与 resampler 的 query 数无关
        # resampler 的 12 个 query 输出会在 calli_align 中被展平为 reference_embeds
        print("Keep num_image_token = 3 (VIT feature patches)")

        del old_resampler
        gc.collect()
        torch.cuda.empty_cache()
        print("Successfully loaded Caoshu CalliAlign")
    else:
        print("Using original CalliAlign")

    detect_model = YOLO(str(YOLO_CHECKPOINT))
    generation_config = dict(num_beams=1, max_new_tokens=1024, do_sample=False)

    return model, tokenizer, detect_model, generation_config


def main():
    parser = argparse.ArgumentParser(description="草书 CalliReader 推理")
    parser.add_argument('--tgt', type=str, required=True, help='识别目标：图片路径或目录')
    parser.add_argument('--prompt', type=str, default='这幅书法作品内容是什么？', help='识别提示词')
    parser.add_argument('--save_name', type=str, default='caoshu_recognition.json', help='结果保存文件名')

    parser.add_argument('--caoshu_ckpt', type=str,
                        default='/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best_val_1.pt',
                        help='草书模型 checkpoint 路径（不指定则使用原模型）')
    parser.add_argument('--num_layers', type=int, default=8, help='Resampler 层数')
    parser.add_argument('--num_learns', type=int, default=12, help='Query 数量')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 率')

    parser.add_argument('--use_p', type=bool, default=True)
    parser.add_argument('--hard_vq', type=bool, default=False)
    parser.add_argument('--drop_zero', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    args = parser.parse_args()

    # 加载模型
    model, tokenizer, detect_model, generation_config = load_model_with_caoshu_ckpt(
        caoshu_ckpt_path=args.caoshu_ckpt if args.caoshu_ckpt else None,
        num_layers=args.num_layers,
        num_learns=args.num_learns,
        dropout=args.dropout
    )

    # 判断是单图还是目录
    if os.path.isfile(args.tgt):
        print("单图识别模式")
        single_rec(
            model, tokenizer, detect_model, generation_config,
            args.tgt, args.prompt,
            args.use_p, args.hard_vq, args.drop_zero,
            args.repetition_penalty, args.verbose
        )
    elif os.path.isdir(args.tgt):
        print("批量识别模式")
        os.makedirs('results', exist_ok=True)
        folder_rec(
            model, tokenizer, detect_model, generation_config,
            args.tgt, args.prompt,
            os.path.join('results', args.save_name),
            args.use_p, args.hard_vq, args.drop_zero,
            args.repetition_penalty, args.verbose
        )
    else:
        raise ValueError(f"目标必须是图片文件或目录: {args.tgt}")


if __name__ == '__main__':
    main()
