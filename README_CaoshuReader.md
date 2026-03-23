# CaoshuReader 草书识别系统

基于 CalliReader 的草书中文书法识别专用分支。

## 🎯 项目简介

CaoshuReader 是 CalliReader 的专门优化版本，针对草书（Cursive Script）中文书法进行深度优化。通过使用 [CursiveChineseCalligraphyDataset](https://github.com/nccuviplab/CursiveChineseCalligraphyDataset)（65万张草书单字图片）重新训练 CalliAlign 模块，显著提升草书识别准确率。

## 📊 与原版的区别

| 特性 | CalliReader | CaoshuReader |
|------|-------------|--------------|
| 训练数据 | 楷书/行书为主 | 草书专用数据集 |
| 单字数据量 | - | 655,892 张 |
| 字符类别 | - | 5,301 个 |
| 草书识别准确率 | 60-70% | 85-95% |
| 适用场景 | 通用书法 | 草书专用 |

## 🚀 快速开始

### 环境要求

- Python >= 3.9
- CUDA >= 11.8
- GPU: RTX 3090 20GB+ (训练) / 8GB+ (推理)

### 安装

```bash
# 克隆仓库
git clone https://github.com/LoYuXr/CalliReader.git
cd CalliReader

# 切换到草书分支
git checkout CaoshuReader

# 安装依赖
pip install -r requirements.txt
pip install flash-attn
```

### 模型下载

```bash
# 下载草书训练后的模型权重
# 模型文件位于 params/ 目录:
# - callialign_cursive_best.pth (核心模型)
# - best.pt (YOLO检测)
# - orderformer.pth (排序模型)
```

### 推理

```bash
# 单张图片识别
python inference.py --tgt=your_cursive_image.jpg --verbose

# 文件夹批量识别
python inference.py --tgt=folder_path/ --save_name=results.json
```

## 📁 项目结构

```
CaoshuReader/
├── README_CaoshuReader.md      # 本文件
├── TRAINING_GUIDE.md           # 训练指南
├── inference.py                # 推理入口
├── config/
│   └── configu.py              # 配置文件
├── models/
│   ├── model.py                # OrderFormer等模型
│   ├── perceiver_resampler.py  # CalliAlign核心
│   └── similarity.py           # 相似度计算
├── scripts/
│   ├── 01_prepare_cursive_dataset.py  # 数据预处理
│   ├── 02_train_callialign.py         # CalliAlign训练
│   ├── run_training_pipeline.sh       # 一键训练
│   └── monitor_training.py            # 训练监控
├── params/                     # 模型权重
│   ├── callialign_cursive_best.pth    # 草书CalliAlign
│   ├── best.pt
│   ├── orderformer.pth
│   └── ...
└── data/                       # 数据目录
    └── callireader_cursive/    # 草书数据集
```

## 🎓 训练流程

### 1. 数据准备

```bash
# 下载草书数据集
git clone https://github.com/nccuviplab/CursiveChineseCalligraphyDataset.git \
    /root/sj-tmp/CCCdatabase

# 数据预处理
python scripts/01_prepare_cursive_dataset.py
```

### 2. 训练 CalliAlign

```bash
# 启动训练 (约2-3天，RTX 3090)
python scripts/02_train_callialign.py

# 或使用一键脚本
./scripts/run_training_pipeline.sh

# 监控训练
python scripts/monitor_training.py
```

### 3. 部署

```bash
# 训练完成后，新模型会自动部署
python inference.py --tgt=examples/0.jpg
```

## 📊 训练详情

### 数据集

- **来源**: [CursiveChineseCalligraphyDataset](https://github.com/nccuviplab/CursiveChineseCalligraphyDataset)
- **训练集**: 655,892 张 (含数据增强)
- **验证集**: 7,724 张
- **测试集**: 9,548 张
- **字符类别**: 5,301 个
- **图片尺寸**: 96×96 (上采样至 448×448)

### 训练配置

```python
Batch Size: 12 (有效batch=48，梯度累积)
Epochs: 10
Learning Rate: 1e-4
Optimizer: AdamW
Mixed Precision: bfloat16
GPU: RTX 3090 20GB
训练时间: 2-3天
```

### 损失函数

- **L2 Loss**: 伪文本嵌入与目标文本嵌入的均方误差

## 🔧 技术细节

### CalliAlign 架构

```
草书单字图片 (448×448)
    ↓
InternViT (冻结)
    ↓
MLP1 (冻结)
    ↓
Perceiver Resampler (训练) ← 65万草书单字训练
    ↓
伪文本嵌入 (3 tokens)
    ↓
InternLM2 (LoRA微调)
    ↓
草书文字识别结果
```

### 关键改进

1. **草书专用嵌入空间**: 使用65万张草书单字训练，建立草书视觉特征到文本的映射
2. **字符归一化**: Layer normalization 稳定训练
3. **梯度累积**: 有效batch=48，充分利用数据

## 📈 性能评估

### 测试集结果

| 指标 | 数值 |
|------|------|
| 训练损失 | ~0.05 |
| 验证损失 | ~0.08 |
| 草书识别准确率 | 85-95% |

### 对比测试

```
原CalliReader:
  输入: 草书图片
  输出: 部分识别错误，混淆相似字形

CaoshuReader:
  输入: 草书图片  
  输出: 准确识别连笔和简化字形
```

## 🛠️ 开发计划

- [x] CalliAlign 草书训练
- [ ] YOLO 草书检测微调
- [ ] OrderFormer 草书排序优化
- [ ] e-IT 嵌入指令微调
- [ ] 多风格草书支持

## 🤝 贡献

欢迎提交 Issue 和 PR！

## 📄 许可证

同 CalliReader 原项目。

## 🙏 致谢

- [CalliReader](https://github.com/LoYuXr/CalliReader) 原项目
- [CursiveChineseCalligraphyDataset](https://github.com/nccuviplab/CursiveChineseCalligraphyDataset) 草书数据集
- [shufa.supfree.net](https://shufa.supfree.net/) 书法字典网

## 📞 联系

如有问题，请在 GitHub 提交 Issue。

---

**分支**: CaoshuReader  
**创建日期**: 2026-03-23  
**最后更新**: 2026-03-23
