# CaoshuReader 训练指南（实际环境版）

> 本文档基于 `/workspace/CalliReader` 的实际代码和 `/root/sj-tmp/datasets/CCC` 数据整理，纠正了旧文档中的路径和脚本错误。

---

## 📁 实际项目结构

```text
/workspace/CalliReader/
├── inference.py              # 推理入口（完整 CalliReader）
├── config/
│   └── configu.py            # 配置（路径、超参）
├── models/
│   ├── model.py              # VIT / MLP1 / Resampler / Tokenizer 加载器
│   ├── perceiver_resampler.py
│   └── similarity.py
├── caoshu/                   # ⭐ 草书训练相关（旧文档中 scripts/ 不存在）
│   ├── train.py              # 主训练脚本（CalliAlign）
│   ├── train_finetune.py     # 微调脚本（从 checkpoint 继续，低 LR）
│   ├── dataset.py            # CaoshuDataset + 数据增强
│   ├── prepare_ccc_split.py  # CCC 数据划分（8:1:1）
│   ├── fix_data_split.py     # 旧 Cursive...Dataset 划分修复
│   ├── analyze_data_quality.py
│   ├── eval_accuracy.py
│   └── pipeline.py           # 草书推理 pipeline
├── utils/
│   └── utils.py
└── params -> /root/sj-tmp/checkpoints/CalliReader/params   # 软链接
```

**关键差异说明**（与旧文档对比）：
- ❌ 旧文档中的 `scripts/01_prepare_cursive_dataset.py`、`scripts/02_train_callialign.py` 等**不存在**
- ✅ 实际训练脚本在 `caoshu/` 目录下，使用 `train.py`
- ✅ 数据实际路径为 `/root/sj-tmp/datasets/CCC`（不是 `/root/sj-tmp/CCC` 或 `CCCdatabase`）
- ✅ 模型基础组件（VIT、MLP1、Tokenizer）在 `/root/sj-tmp/checkpoints/CalliReader/params/`

---

## 📊 数据概况

| 数据集 | 路径 | 大小 | 说明 |
|---|---|---|---|
| CCC（增强后） | `/root/sj-tmp/datasets/CCC/` | 6.8 GB | 10,423 字，414,807 张图（90,877 原图 + 323,930 增强） |
| CCC_split | `/root/sj-tmp/datasets/CCC_split/` | ~13 GB | 划分后的训练/验证/测试集 |
| 旧草书数据集 | `/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/` | 5.3 GB | 原始数据集，含 Training/Validation 分层 |

---

## 🚀 快速开始

### 步骤 0：环境准备（只需执行一次）

```bash
cd /workspace/CalliReader

# 1) 划分 CCC 数据为 Training/Validation/Test（8:1:1）
python caoshu/prepare_ccc_split.py

# 输出：
#   /root/sj-tmp/datasets/CCC_split/Training/
#   /root/sj-tmp/datasets/CCC_split/Validation/
#   /root/sj-tmp/datasets/CCC_split/Test/
#   /workspace/CalliReader/params -> /root/sj-tmp/checkpoints/CalliReader/params

# 2) 检查 params 软链接是否正确
ls -la params/
# 应看到 vit_model.pt、 mlp1.pth、 token_embedding.pth 等
```

### 步骤 1：训练 CalliAlign

```bash
cd /workspace/CalliReader/caoshu

python train.py \
  --data_root /root/sj-tmp/datasets/CCC_split \
  --save_dir /root/sj-tmp/checkpoints/CaoshuReader \
  --batch_size 32 \
  --grad_accum 16 \
  --lr 1e-4 \
  --total_steps 100000 \
  --warmup_steps 10000 \
  --num_layers 8 \
  --num_learns 12 \
  --dropout 0.1 \
  --label_smoothing 0.1 \
  --val_every 2000 \
  --save_every 5000 \
  --keep_ckpts 5 \
  --keep_best_val 3 \
  --lr_patience 6 \
  --lr_factor 0.3
```

**显存参考**（单卡 RTX 3090，**24 GB**）：
- `batch_size=32` + `grad_accum=16` + 输入 `224×224` → 有效 batch = 512，显存约 **13 GB**
- 由于 224×224 输入下 activation 极小，batch_size 从 16 翻倍到 32 显存几乎不变
- 24GB 显存下非常安全，还有约 **11 GB** 余量
- 如显存不足，改为 `batch_size=16`，`grad_accum=16`（有效 batch=256）

### 步骤 2：恢复训练（注意）

> ⚠️ 当前环境因 PyTorch 版本升级（2.8.0），从旧 checkpoint resume 时加载 `optimizer_state_dict` 会导致卡死。如需继续训练，建议**从头开始**（100k 步仅需约 10 小时，速度远快于预期）。

若必须从 checkpoint 恢复模型权重（不恢复 optimizer）：

```bash
cd /workspace/CalliReader/caoshu
python train_finetune.py \
  --data_root /root/sj-tmp/datasets/CCC_split \
  --save_dir /root/sj-tmp/checkpoints/CaoshuReader \
  --resume /root/sj-tmp/checkpoints/CaoshuReader/caoshu_best.pt \
  --lr 1e-5 --total_steps 100000
```

### 步骤 3：推理测试

训练的是 **CalliAlign（PerceiverResampler）**，用于将草书图像对齐到文本嵌入空间。
完整 CalliReader 推理（含检测+排序+识别）：

```bash
cd /workspace/CalliReader
python inference.py --tgt examples/0.jpg --verbose
```

---

## ⚙️ 训练参数详解（train.py）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_root` | `.../Cursive_Chinese_Calligraphy_Dataset` | 数据集根目录，需含 `Training/`、`Validation/` 子目录 |
| `--save_dir` | `/root/sj-tmp/checkpoints/CaoshuReader` | checkpoint 保存路径 |
| `--batch_size` | 16 | 单步 batch size |
| `--grad_accum` | 16 | 梯度累积步数，有效 batch = batch_size × grad_accum |
| `--lr` | 1e-4 | 初始学习率 |
| `--total_steps` | 100000 | 总训练步数 |
| `--warmup_steps` | 10000 | 线性 warmup 步数（论文建议 10k） |
| `--num_layers` | 8 | PerceiverResampler 层数（论文：8） |
| `--num_learns` | 12 | Learnable query 数量（论文：12） |
| `--dropout` | 0.1 | Dropout 率 |
| `--label_smoothing` | 0.1 | Label smoothing 系数 |
| `--val_every` | 2000 | 每 N 步在验证集评估一次 |
| `--save_every` | 5000 | 每 N 步保存常规 checkpoint |
| `--keep_ckpts` | 5 | 保留最近 N 个 step checkpoint（每个约 3.2GB） |
| `--keep_best_val` | 3 | 保留验证 loss 最低的 top-K 模型 |
| `--lr_patience` | 6 | 验证 loss 连续 N 次不下降则降低 LR |
| `--lr_factor` | 0.3 | LR 缩减倍数 |
| `--min_lr` | 1e-7 | LR 下限 |
| `--early_stop` | None | 连续 N 次验证无改善则早停（默认不启用） |
| `--resume` | None | 从 checkpoint 恢复训练 |
| `--skip_disk_check` | False | 跳过磁盘空间检查 |

---

## 💾 Checkpoint 说明

在 `/root/sj-tmp/checkpoints/CaoshuReader/` 下会生成：

| 文件 | 说明 |
|---|---|
| `caoshu_step{N}.pt` | 常规 checkpoint（每 `save_every` 步） |
| `caoshu_best.pt` | 训练 loss 最低的模型 |
| `caoshu_best_val_1.pt` | 验证 loss 第 1 名的模型 |
| `caoshu_best_val_2.pt` | 验证 loss 第 2 名的模型 |
| `caoshu_best_val_3.pt` | 验证 loss 第 3 名的模型 |
| `caoshu_final.pt` | 训练结束时的最终模型 |

**自动清理**：当常规 checkpoint 超过 `keep_ckpts` 个时，自动删除最旧的 `caoshu_step*.pt`，**不会**删除 `best` 和 `final`。

**磁盘保护**：
- 剩余空间 < 5GB：`emergency`，跳过保存
- 剩余空间 < 10GB：`critical`，只保存 best_val
- 剩余空间 > 20GB：`normal`，正常保存

---

## 🔍 监控训练

### 方式 1：进入 screen 窗口（推荐）
```bash
screen -r caoshu_train      # 进入训练窗口
# 退出（训练继续）: 按 Ctrl+A，再按 D
```

### 方式 2：查看日志
```bash
tail -f /root/sj-tmp/checkpoints/CaoshuReader/train.log
```

### 方式 3：GPU 监控
```bash
watch -n 1 nvidia-smi
```

### 方式 3：验证集指标
`train.py` 每 `val_every` 步自动在验证集上计算 `alignment_loss`，并输出：
```text
[Val] step=  2000 val_loss=0.2341 | disk: ✅ 145.3GB
```

---

## 🛠️ 故障排除

### 问题 1：`FileNotFoundError: ./params/vit_model.pt`

**原因**：`configu.py` 中使用的是相对路径 `./params/...`，运行目录不对或软链接未创建。

**解决**：
```bash
cd /workspace/CalliReader
# 检查软链接
ls -la params/
# 如果不存在，手动创建
ln -s /root/sj-tmp/checkpoints/CalliReader/params params
```

### 问题 2：显存不足 (OOM)

**解决**：调小 `batch_size`，增大 `grad_accum`，保持有效 batch 不变：
```bash
python train.py ... --batch_size 8 --grad_accum 32
```

### 问题 3：训练中断 / 恢复训练

> ⚠️ **resume 当前不可用**：PyTorch 2.8.0 与旧 checkpoint 的 `optimizer_state_dict` 不兼容，加载后训练会在 step ~1990 附近卡死。

**解决**：从头重新训练（速度约 3 步/秒，100k 步约 10 小时）。如需保留已训练进度，可将旧 checkpoint 作为预训练权重：
```bash
python train.py \
  --resume /root/sj-tmp/checkpoints/CaoshuReader/caoshu_step50000.pt \
  --total_steps 100000   # 只加载 model_state_dict，重建 optimizer
```

### 问题 4：验证集无法加载

**原因**：CCC 原始数据没有 `Validation` 目录，或划分不正确。

**解决**：重新运行数据划分脚本：
```bash
rm -rf /root/sj-tmp/datasets/CCC_split
python caoshu/prepare_ccc_split.py
```

### 问题 5：磁盘空间不足

**解决**：
- 减小 `--keep_ckpts`（如改为 3）
- 手动删除旧的 `caoshu_step*.pt`
- 或使用 `--skip_disk_check`（不推荐，可能打满磁盘）

---

## 📈 预期结果

| 指标 | 预期范围 |
|---|---|
| 训练 loss | warmup 期 ~1.0（正常，lr 极低），step>10000 后快速下降，最终 ~0.05 |
| 验证 loss | 稳定在 ~0.08 以下 |
| 训练时间 | 约 5 小时（单卡 RTX 3090 24GB，224×224 输入，~5 步/秒） |

**已有 checkpoint**（截至 2026-04）：
- `/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best_val_1.pt`（约 3.2GB）
- `/root/sj-tmp/checkpoints/CaoshuReader/caoshu_final.pt`

---

## 📝 后续步骤

1. **e-IT 微调**：使用 CalliTrain 数据对 InternVL 进行嵌入指令微调
2. **YOLO 微调**：如检测效果不佳，用合成的页面数据微调 `params/best.pt`
3. **数据增强**：已有 32 万张增强图，可视情况继续增加变体

---

**最后更新**：2026-05-07
