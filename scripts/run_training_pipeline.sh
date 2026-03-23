#!/bin/bash
# CalliReader 草书训练完整流程
# 针对 RTX 3090 20GB 优化

set -e  # 遇到错误立即退出

echo "=================================================="
echo "CalliReader 草书训练流程"
echo "=================================================="
echo "开始时间: $(date)"
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 设置环境变量
export PYTHONPATH="/workspace/CalliReader:/workspace/CalliReader/InternVL:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
mkdir -p /workspace/CalliReader/logs

# ========== 步骤 0: 检查数据集 ==========
echo ""
echo "=================================================="
echo "步骤 0: 检查数据集"
echo "=================================================="

CCC_DATASET="/root/sj-tmp/CCCdatabase_tmp"
if [ ! -d "$CCC_DATASET" ]; then
    echo "错误: 找不到草书数据集: $CCC_DATASET"
    echo "请先下载: git clone https://github.com/nccuviplab/CursiveChineseCalligraphyDataset.git $CCC_DATASET"
    exit 1
fi

echo "✓ 数据集已存在: $CCC_DATASET"

# ========== 步骤 1: 数据预处理 ==========
echo ""
echo "=================================================="
echo "步骤 1: 数据预处理"
echo "=================================================="

if [ -f "/workspace/CalliReader/data/callireader_cursive/char_mapping.json" ]; then
    echo "✓ 数据已预处理，跳过此步骤"
else
    echo "运行数据预处理脚本..."
    cd /workspace/CalliReader
    python scripts/01_prepare_cursive_dataset.py 2>&1 | tee logs/01_prepare_data.log
    
    if [ ! -f "/workspace/CalliReader/data/callireader_cursive/char_mapping.json" ]; then
        echo "错误: 数据预处理失败"
        exit 1
    fi
fi

# ========== 步骤 2: 训练 CalliAlign ==========
echo ""
echo "=================================================="
echo "步骤 2: 训练 CalliAlign (约2-3天)"
echo "=================================================="

echo "开始训练 CalliAlign..."
echo "此步骤将训练约 2-3 天，可以随时中断并恢复"
echo "按 Ctrl+C 中断，重新运行脚本会自动恢复"

cd /workspace/CalliReader
python scripts/02_train_callialign.py 2>&1 | tee logs/02_train_callialign.log

# 检查训练结果
if [ ! -f "/workspace/CalliReader/params/callialign_cursive_best.pth" ]; then
    echo "错误: CalliAlign 训练失败"
    exit 1
fi

echo "✓ CalliAlign 训练完成"

# ========== 步骤 3: 更新配置 ==========
echo ""
echo "=================================================="
echo "步骤 3: 更新模型配置"
echo "=================================================="

# 备份原模型
if [ ! -f "/workspace/CalliReader/params/callialign_backup.pth" ]; then
    cp /workspace/CalliReader/params/callialign.pth /workspace/CalliReader/params/callialign_backup.pth
    echo "✓ 备份原 CalliAlign 模型"
fi

# 使用新模型
cp /workspace/CalliReader/params/callialign_cursive_best.pth /workspace/CalliReader/params/callialign.pth
echo "✓ 更新 CalliAlign 为草书训练版本"

# ========== 步骤 4: 测试 ==========
echo ""
echo "=================================================="
echo "步骤 4: 测试新模型"
echo "=================================================="

cd /workspace/CalliReader

echo "测试示例图片..."
python inference.py --tgt=/workspace/CalliReader/examples/0.jpg --verbose 2>&1 | tee logs/04_test.log

echo ""
echo "=================================================="
echo "训练流程完成!"
echo "=================================================="
echo "结束时间: $(date)"
echo ""
echo "新模型已部署，可以使用以下命令测试:"
echo "  python inference.py --tgt=your_image.jpg"
echo ""
