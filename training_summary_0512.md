# 书法识别项目训练总结（2026-05-12）

> 本文件合并自 `TRAINING_SUMMARY.md`、`log0510.md` 和 `summary0510.md`，去重并按逻辑重新组织。

---

## 一、项目概述

`/caoshu/` 项目包含 **两个模型**，形成一套完整的书法识别 pipeline：

| 模型 | 用途 | 训练数据 | 检查点位置 |
|------|------|---------|-----------|
| **CalliReader** | 通用书法VLM（整页OCR、理解、问答） | `/root/sj-tmp/callireader_models` | `InternVL/` + `params/`（软链接） |
| **CaoshuReader** | 草书单字识别（基于CalliReader微调） | `/root/sj-tmp/datasets/CaoshuMerged` | `/root/sj-tmp/checkpoints/CaoshuReader_v2/` |

**CaoshuReader** 是基于 **CalliReader** 微调出来的草书专用模型，只训练 `PerceiverResampler`，冻结其他所有组件。

**输入**：一张书法作品的整图扫描件  
**输出**：识别出图中每一个草书单字，并按阅读顺序排列成文本

**核心架构（分离式 Pipeline）：**

```
[图片] → YOLO分割(best.pt) → [单字图片]
    ↓
Vision Model(vit_model.pt) → MLP1(mlp1.pth) → [视觉特征]
    ↓
Resampler(callialign.pth) → [字符embedding]
    ↓
与预计算字符embedding匹配 → Top-K候选字
    ↓
后处理（阅读顺序排序 + 可选LLM纠错）→ [最终文本]
```

**模块说明：**

| 模块 | 参数规模 | 角色 | 状态 |
|------|---------|------|------|
| **InternViT** | ~580M | 视觉特征提取器，输入 448×448 | 冻结 |
| **MLP1** | ~65M | 视觉投影层，投影到 4096 维 | 冻结 |
| **PerceiverResampler** | **0.571B** | 可训练核心，4层×3 queries | 🔄 **正在重训** |
| **LLM Token Embedding** | ~724M | 冻结文本嵌入层 | 冻结 |
| **YOLO** | — | 整图单字检测 | 冻结 |

**损失函数**（alignment loss）：
```python
tgt_norm = F.normalize(target_embed, dim=-1).unsqueeze(1)
pred_norm = F.normalize(pred, dim=-1)
cosine_sim = (pred_norm * tgt_norm).sum(dim=-1)
loss = (1 - cosine_sim).mean()
```

---

## 二、组件状态

### 2.1 核心组件

| 组件 | 权重文件 | 大小 | 状态 | 说明 |
|------|----------|------|------|------|
| YOLO分割 | `params/best.pt` | 62MB | ✅ | 分割单字，可用 |
| Vision Model | `params/vit_model.pt` | 580MB | ✅ | 提取视觉特征，冻结 |
| MLP1 | `params/mlp1.pth` | 65MB | ✅ | 特征降维，冻结 |
| **Resampler** | `params/callialign.pth` | **3.2GB** | 🔄 | **核心瓶颈，正在重训** |
| Token Embeddings | `params/token_embedding.pth` | 724MB | ✅ | 字符embedding，冻结 |
| Gauss Norm Embedding | `params/gauss_norm.pth` | 724MB | ✅ | 归一化token embedding |
| New1000 Token Embedding | `params/new1000_token_embedding.pth` | 724MB | ✅ | 扩展token embedding |
| OrderFormer | `params/orderformer.pth` | 26MB | ✅ | 框排序Transformer |
| e-IT LoRA | `outputs/eit_simple_overfit/final` | 2.7GB | ⚠️ | 辅助纠错，非主力 |

### 2.2 当前训练检查点

```
/root/sj-tmp/checkpoints/CaoshuReader_v2/        (当前主力)
├── caoshu_best.pt        (3.2GB)  最佳模型 (step 49761, loss 0.1751)
├── caoshu_step40000.pt   (3.2GB)
├── caoshu_step45000.pt   (3.2GB)
└── caoshu_step50000.pt   (3.2GB)  当前最新

/root/sj-tmp/checkpoints/CaoshuReader_lr5e5/     (历史实验)
└── caoshu_best.pt        (3.2GB)  step 20605, loss 0.590
```

---

## 三、训练历程

### 3.1 Resampler 初训（callialign.pth）

| 配置 | 数值 |
|------|------|
| 数据集 | Cursive_Chinese_Calligraphy_Dataset |
| 训练样本 | **6,493** |
| 验证样本 | 4,281 |
| 训练步数 | 50,000 |
| Best Loss | 0.086 |
| **Validation Top-1** | **29%** |
| **Validation Top-5** | 约 50% |

**结论**：655,742 样本（正楷+草书混合）不足以学好纯草书分类，Resampler 欠拟合，29% Top-1 是主要瓶颈。

---

### 3.2 lr=1e-4 训练实验（已停止）

运行至 step 13000+ 后停止。

| 指标 | 数值 | 说明 |
|------|------|------|
| Best Loss | **0.6634 @ step 51** | 极低步数即达到最佳，之后未再下降 |
| 当前 Loss | 0.73 ~ 0.85 | step 10000+ 后持续震荡 |
| 显存占用 | ~8 GB / 23.68 GB | FlashAttention 启用后 |

**问题判断**：❌ **Loss 在 step 51 后不再下降，持续震荡** → 学习率过高（`lr=1e-4`）。

---

### 3.3 lr=5e-5 训练实验（CaoshuReader_lr5e5）

**启动命令**：
```bash
cd /caoshu && conda activate caoshu && python caoshu/train.py \
    --data_root=/root/sj-tmp/datasets/CCC_split \
    --save_dir=/root/sj-tmp/checkpoints/CaoshuReader_lr5e5 \
    --total_steps=100000 --batch_size=16 --grad_accum=16 --lr=5e-5
```

该实验产出 `caoshu_best.pt` (step 20605, loss 0.590)，但 **非当前主力**。旧 checkpoint 目录 `/root/sj-tmp/checkpoints/CaoshuReader_lr5e5/` 已清理，仅保留 `caoshu_best.pt`（3.2GB）。

---

### 3.4 e-IT LoRA 微调（language_model）

试图用 539 个纯草书样本微调 InternVL 的 LLM，让模型学会"读"草书文本。

| 配置 | 数值 |
|------|------|
| 数据 | 539 个 e-IT 样本（预计算 embedding） |
| 微调范围 | 仅 LLM，LoRA r=128, alpha=256 |
| 目标模块 | wqkv, wo, w1, w2, w3 |
| 优化器 | AdamW8bit |
| Epoch | 10 |
| 最终 Loss | 0.0591 |

**端到端测试结果：**

| 图片 | 基线输出 | LoRA 输出 |
|------|----------|-----------|
| 2.jpg | 重复"云飞远天"80次 | 雪龙远飞天马行地尘色年 |
| 6.jpg | 问陵墓问题 | 国破山河花柳春柔... |
| 10.jpg | 清风明月本无价... | 清风明月本无心... |

**结论**：
- ✅ 基线 InternVL 完全不会读草书
- ✅ LoRA 治好了重复问题，能输出相关内容
- ❌ 字符串匹配准确率 0%，实际内容命中率约 80%

**根本原因**：Visual 路径不兼容
```
e-IT 训练：图片 → callialign.pth(Resampler) → [UNUSED_TOKEN_140] → LoRA LLM
端到端测试：图片 → InternVL原始Resampler → <IMG_CONTEXT> → LoRA LLM
两个 Resampler 提取的视觉特征完全不同 → LoRA 知识无法迁移
```

---

### 3.5 Resampler 大数据集恢复训练（🔄 当前主力）

#### 3.5.1 数据集合并

| 数据集 | Training 样本 | 字符数 | 说明 |
|--------|--------------|--------|------|
| CCC_split | **338,870** | 8,398 | 纯草书，主要来源 |
| Original (Cursive_Chinese_Calligraphy_Dataset) | **655,742** | 5,300 | 正楷+草书混合 |
| V2 | **300,743** | 2,353 | 补充数据 |
| **合并后** | **974,113** | **8,398** | 符号链接合并，不复制 |

合并路径: `/root/sj-tmp/datasets/CaoshuMerged/Training`  
验证集: `/root/sj-tmp/datasets/CCC_split/Validation` (61,181 样本)

#### 3.5.2 训练配置

- 从 `callialign.pth` 恢复
- batch_size=16, grad_accum=4, effective_batch=64
- lr=5e-5
- 目标 100,000 步
- save_every=5000, keep_ckpts=3
- 输出: `/root/sj-tmp/checkpoints/CaoshuReader_v2`

#### 3.5.3 验证准确率里程碑

| Step | Validation Top-1 | Validation Top-5 | 备注 |
|------|-----------------|------------------|------|
| 5,000 | 23.30% | 39.05% | 恢复起点 |
| 10,000 | 34.01% | 54.44% | — |
| 15,000 | 36.95% | 57.62% | — |
| 20,000 | 37.32% | 58.29% | — |
| 25,000 | 36.77% | 57.35% | 短暂回落 |
| 30,000 | 41.07% | 62.57% | **突破 40%** |
| 35,000 | 44.51% | 66.50% | — |
| 40,000 | 46.23% | 67.95% | — |
| 45,000 | 47.50% | 69.44% | — |
| **46,000** | **47.66%** | **69.53%** | **新 best loss 0.1786** |
| **49,000** | **47.79%** | **69.81%** | — |
| 76,000 | 49.35% | 71.52% | **突破 49%** |
| 80,000 | 50.46% | 71.88% | **突破 50%** |
| 84,000 | 51.38% | 72.92% | — |
| 88,000 | 52.29% | 73.62% | — |
| **93,000** | **53.19%** | **74.42%** | **当前最佳 Top-1** |

#### 3.5.4 核心对比

| 模型 | 训练样本 | 验证集 | Top-1 | Top-5 |
|------|---------|--------|-------|-------|
| callialign.pth（旧） | **655,742** | 7,724（小） | **29%** | ~50% |
| **新模型 @ step 93,000** | **974,113** | **61,181（大）** | **53.19%** | **74.42%** |
| **提升幅度** | — | — | **+24.2%** | **+24.4%** |

> 注：53.19% 是在 61,181 大验证集上的结果，比原来 7,724 小验证集的 29% 含金量高得多。

#### 3.5.5 整图识别流程（`pipeline.py`）

```
输入整图书法
  → YOLO 检测出每个字的位置框
  → 按阅读顺序排序（从上到下、从左到右）
  → 逐个裁剪为单字图像 (224×224)
  → ViT + MLP1 提取视觉特征
  → PerceiverResampler 预测 embedding
  → 与预计算的字符 embedding 做余弦相似度
  → 输出 Top-3 候选字 + 置信度
  → 合并为完整识别文本
```

### 3.5.6 当前状态

- **训练任务**：正在后台运行（step 94,000/100,000）
- **最新 checkpoint**：`caoshu_step90000.pt`
- **Best checkpoint**：`caoshu_best.pt`（自动跟踪最低 loss）
- **Best Loss**：**0.1400** @ step 91877
- **最新 Validation**：Top-1 **53.19%** @ step 93000
- **磁盘空间**：57GB 可用（已删除 CCC 原始数据集释放 6.8GB）
- **预计完成**：剩余 ~6,000 步
- **架构图**：已生成 `/caoshu/caoshu.jpg`
- **Pipeline 修复**：去掉推理时的 `.resize((224, 224))`，transform 直接到 448×448

---

## 四、关键教训

1. **数据量决定上限**：655K 样本 → 29%，974K 样本 → 53.19%（目标 55%+）
2. **Visual 路径必须一致**：训练和推理用不同 Resampler = 白训
3. **Loss 低不等于效果好**：e-IT Loss 0.059 但实际幻觉严重
4. **验证集规模影响指标观感**：大验证集（61K）比小验证集（4K）更真实可靠
5. **磁盘管理**：每个 ckpt 3.2GB，需自动清理机制
6. **学习率敏感**：lr=1e-4 导致 loss 触底反弹，lr=5e-5 才能持续下降

---

## 五、当前策略

```
┌──────────────────────────────────────────┐
│  主攻：Resampler 大数据量训练             │
│  当前：47.79% → 目标 50%+                │
│  辅助：e-IT LoRA 后处理纠错（非优先）     │
│  不搞：端到端 InternVL + LoRA（已证伪）   │
└──────────────────────────────────────────┘
```

### 为什么保持分离架构？

端到端 InternVL + LoRA 的问题：
- ❌ Visual 路径不兼容
- ❌ 需要重新训练 Resampler 与 InternVL 匹配
- ❌ 539 样本训 LLM 严重过拟合

分离架构的优势：
- ✅ Resampler 直接输出 embedding，可解释性强
- ✅ YOLO 分割 + 逐字识别，错误可定位
- ✅ 可以单独优化每个环节

### 未来增强方向

1. **Resampler 继续训练**（当前重点）
2. **LLM 后处理纠错**：Top-3 → 语义选择
3. **数据增强**：旋转、模糊、对比度变化
4. **阅读顺序优化**：字序排列算法改进

---

## 六、项目目录结构

```
/caoshu/
├── InternVL/          → 软链接 → /root/sj-tmp/callireader_models/InternVL
├── params/            → 软链接 → /root/sj-tmp/callireader_models/params
├── caoshu/
│   ├── dataset.py          CCC_split数据加载
│   ├── pipeline.py         整图草书识别Pipeline
│   ├── train.py            草书训练脚本
│   ├── train_finetune.py   微调脚本
│   ├── visualizer.py       YOLO可视化
│   └── ...
├── models/
│   ├── model.py            模型加载函数
│   ├── perceiver_resampler.py  Perceiver架构
│   ├── get_embeds.py       嵌入提取工具
│   └── similarity.py       VQ + Loss函数
├── config/
│   └── configu.py          全局配置
├── utils/
│   └── utils.py            图像处理、工具函数
├── train/                  CalliReader原始训练代码 (xtuner)
├── inference.py            CalliReader推理入口
├── evaluate.py             CalliBench评测
├── test_caoshu.py          草书模型测试
└── requirements.txt
```

## 七、代码与数据集状态

### 7.1 代码状态

| 文件 | 状态 |
|------|------|
| `caoshu/train.py` | ✅ 可运行，支持 resume，已修复 module. 前缀 |
| `caoshu/train_finetune.py` | ✅ 微调脚本（从checkpoint继续，降低lr） |
| `test_caoshu.py` | ✅ 可运行，自推断 + dtype 转换 |
| `caoshu/pipeline.py` | ✅ 已修正 token id + 原图裁字 + YOLO imgsz |
| `caoshu/visualizer.py` | ✅ YOLO imgsz 参数已添加 |
| `models/model.py` | ✅ torch.load weights_only 已明确 |
| `inference.py` | ✅ CalliReader单图/文件夹推理入口 |
| `evaluate.py` | ✅ CalliBench评测（整页识别、区域OCR、选择题等） |

### 7.2 数据集状态

| Split | 路径 | 类别数 | 样本数 |
|-------|------|-------|-------|
| Training (合并) | `/root/sj-tmp/datasets/CaoshuMerged/Training` | 8,398 | 974,113 |
| Validation | `/root/sj-tmp/datasets/CCC_split/Validation` | 7,440 | 61,181 |
| Test | `/root/sj-tmp/datasets/CCC_split/Test` | 4,050 | 14,756 |

### 7.3 数据构成与来源

**处理流程**：
1. `CursiveChineseCalligraphyDataset` (91×91 字体原图) + `shufazidian` (528×538 名家真迹，48K 样本，6,602 字)
2. 合并后删除旧合成数据（带 gen 标记的）
3. 因总数不足，统一增强合成，生成新 gen 数据
4. 存放于 `/root/sj-tmp/datasets/CCC`，再 split 为 `CCC_split`
5. `/CCC` 原始全集已删除（`CCC_split` 已完整包含所有数据）

**CCC_split/Training 构成**：

| 类型 | 数量 | 占比 |
|------|------|------|
| 合成数据 (gen) | 264,179 | **78%** |
| 原始数据 | 74,691 | **22%** |
| **总计** | **338,870** | — |

**Original** (655,742) 和 **V2** (300,743) 是另外的独立数据源，合并为 CaoshuMerged (974K)。

---

## 八、问题与修复记录

| # | 问题 | 现象 | 解决方案 |
|---|------|------|---------|
| 1 | OOM（训练/测试） | ViT 激活保留导致显存溢出 | 冻结模块包 `torch.no_grad()` + `autocast(bf16)` |
| 2 | OOM（Pipeline） | YOLO 读取超高分辨率原图 | YOLO `imgsz=1344` + letterbox |
| 3 | DDP `module.` 前缀 | `Unexpected key(s) in state_dict` | 加载前 `k.replace('module.', '')` |
| 4 | pipeline token id 错误 | 预计算 embedding 使用错误索引 | 通过 tokenizer 转为正确 token id |
| 5 | 准确率 0% (`caoshu_best.pt`) | 所有样本预测为同一个字 | `caoshu_best.pt` collapse，`callialign.pth` 可用（29%） |
| 6 | `CaoshuMerged/Validation` 缺失 | FileNotFoundError | 符号链接到 `CCC_split/Validation` |
| 7 | e-IT checkpoint-700 损坏 | 磁盘满导致保存失败 | 删除不完整文件，调整 keep_ckpts |
| 8 | 推理预处理不一致 | pipeline 先 resize 到 224 再转 448，丢失细节 | 去掉中间 224 步骤，transform 直接到 448 |

---

## 九、文件索引

| 文件 | 说明 |
|------|------|
| `training_summary_0512.md` | 本文档（合并版） |
| `TRAINING_SUMMARY.md` | 详细训练总结（历史版） |
| `log0510.md` | 逐日训练日志（历史版） |
| `summary0510.md` | 系统架构总结（历史版） |
| `report_0511.md` | 项目进展报告 |
| `caoshu/train.py` | Resampler 训练脚本 |
| `caoshu/pipeline.py` | 整图识别 Pipeline |
| `caoshu/test_caoshu.py` | 验证集准确率测试 |
| `caoshu/dataset.py` | 数据集加载 |
| `scripts/eit_train_simple.py` | e-IT 初始训练脚本 |
| `scripts/eit_train_resume.py` | e-IT 恢复训练脚本 |
| `scripts/eit_inference_test.py` | e-IT 推理测试脚本 |
| `scripts/e2e_image_test.py` | 端到端图片测试脚本 |
| `scripts/build_eit_dataset.py` | e-IT 数据集构建 |
| `scripts/precompute_samples_embeddings.py` | 样本 embedding 预计算 |
| `scripts/smoke_test_generate.py` | 冒烟测试样本生成 |
| `scripts/merge_datasets.py` | 数据集合并工具 |
| `scripts/resume_resampler_train.sh` | Resampler 恢复训练启动脚本 |

---

## 十、快速参考

### CalliReader 单图识别
```bash
cd /caoshu
python inference.py --tgt=imgs/2.jpg --prompt="这幅书法作品内容是什么？"
```

### 启动 Resampler 训练
```bash
cd /caoshu && bash scripts/resume_resampler_train.sh
```

### 测试单字准确率
```bash
cd /caoshu && conda activate caoshu && python test_caoshu.py \
    --ckpt=/caoshu/params/callialign.pth \
    --data_root=/root/sj-tmp/datasets/CCC_split \
    --split=Validation --num_test=100 --batch_size=16
```

### 整图识别 Pipeline（CaoshuReader）
```bash
cd /caoshu && conda activate caoshu && python caoshu/pipeline.py \
    --image=imgs/2.jpg \
    --ckpt=/caoshu/params/callialign.pth \
    --data_root=/root/sj-tmp/datasets/CCC_split \
    --output=outputs/pipeline --yolo_imgsz=1344
```

### 继续训练草书模型
```bash
cd /caoshu/caoshu
python train.py \
    --data_root=/root/sj-tmp/datasets/CaoshuMerged/Training \
    --save_dir=/root/sj-tmp/checkpoints/CaoshuReader_v2 \
    --resume=/root/sj-tmp/checkpoints/CaoshuReader_v2/caoshu_best.pt \
    --total_steps=150000 \
    --batch_size=16 --grad_accum=4 --lr=5e-5
```

### 加载 e-IT LoRA 推理
```python
from peft import PeftModel
model.language_model = PeftModel.from_pretrained(
    base_language_model,
    "/caoshu/outputs/eit_simple_overfit/final"
)
```

---

*更新时间：2026-05-12*  
*训练状态：Resampler 恢复训练中（step 94,000/100,000）*  
*当前最佳：Top-1 53.19%，Top-5 74.42%，Best Loss 0.1400*  
*磁盘状态：/root/sj-tmp 可用 57GB（已删除 CCC 原始数据集释放 6.8GB）*  
*架构图：/caoshu/caoshu.jpg*
