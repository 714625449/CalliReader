#!/bin/bash
# 恢复 Resampler 训练，使用合并后的更大的数据集

cd /caoshu
source /root/miniconda3/bin/activate caoshu

python caoshu/train.py \
    --data_root=/root/sj-tmp/datasets/CaoshuMerged \
    --split=Training \
    --save_dir=/root/sj-tmp/checkpoints/CaoshuReader_v2 \
    --total_steps=100000 \
    --batch_size=16 \
    --grad_accum=4 \
    --lr=5e-5 \
    --save_every=5000 \
    --keep_ckpts=3 \
    --resume=/caoshu/params/callialign.pth \
    --num_layers=4 \
    --skip_disk_check
