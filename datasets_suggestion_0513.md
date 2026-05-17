# CaoshuReader 数据集优化方案（落地版）

> 生成时间：2026-05-13（v2.3 最终版）  
> 当前基线：Top-1 53.83% / Top-5 74.91%（step 100,000，验证集 80% gen）  
> 最佳模型：caoshu_best.pt @ step 91,877, loss 0.1400 → `/caoshu/params/callialign_v2.pth`  
> ⚠️ **核心认知：真实推理场景 = 100% 真迹，当前 53.83% 是被低估的**

---

## 一、当前数据集全景

### 1.1 数据构成

| 数据集 | 样本数 | 字类 | original% | gen% | 与真实场景的距离 |
|--------|--------|------|-----------|------|------------------|
| **CaoshuMerged/Training** | **974,113** | **10,423** | **72.9%** | **27.1%** | 较近 |
| CCC_split/Training | 338,870 | 10,423 | 22.0% | 78.0% | 远 |
| **CCC_split/Validation** | **61,181** | **9,300** | **19.8%** | **80.2%** | **很远** |
| CCC_split/Test | 14,756 | 4,050 | 27.5% | 72.5% | 远 |
| **shufazidian** | **48,074** | **6,602** | **~100%** | **0%** | **完全一致** |

### 1.2 关键认知

```
真实推理场景（pipeline.py）
    ↓
YOLO 从真迹照片切字 → 100% original，0% gen
    ↓
当前验证集（CCC_split/Val）= 80% gen
    ↓
53.83% 是被【低估】的，不是上限。真实基线可能 60%-70%。
```

---

## 二、执行策略：两条并行 Track

```
Track 1（数据侧）          Track 2（推理侧）
    │                            │
    ▼                            ▼
P0 建真迹验证集              P2b 纯文本 rerank 原型
    │                            │
测现有模型真实基线            现有模型 + LLM rerank
    │                            │
P1 追加 shufazidian           端到端 Pipeline 准确率
    │                            │
再测真迹验证集                与 Track 1 叠加增益
```

**Track 2 最快**：1-2 天就能看到 rerank 能否把 Top-1 往 65%+ 推。与 Track 1 不冲突，可叠加。

---

## 三、Day 0：前置探查（1-1.5 小时，消掉最大不确定性）

在投入 Week 1 之前，先做 5 个探查，每个 5-30 分钟：

### 3.1 探查 1：前置实验 + 配对对照（15 分钟）

**目标**：拿到 shufazidian 全量 Top-1 + CCC_split/Val original 子集 Top-1，对比判断分布差异。

```python
# 实验 A：shufazidian 全量 48K（反色 + resize 448，不做 YOLO）
# 实验 B：CCC_split/Val 中不含 _gen 的 original 子集（~12K 张）
# 同一个模型 callialign_v2.pth 跑两组
```

**结果解读**：

| 场景 | 含义 | 后续调整 |
|------|------|----------|
| A ≈ B（差距 < 3%） | shufazidian 和已有 original 分布接近 | P1 增量有限，重点放 P2b rerank |
| A >> B（差距 > 5%） | shufazidian 是新分布，风格差异大 | P1 紧急度上升 |
| A << B | shufazidian 预处理有问题（如反色错了） | 先修预处理 |

### 3.2 探查 2：颜色空间 sanity check（5 分钟）

**目标**：确认训练集和 shufazidian 的背景色关系，避免反色做反。

```python
from PIL import Image
import numpy as np

# 读一张训练集图
train_img = Image.open('/root/sj-tmp/datasets/CaoshuMerged/Training/一/0.jpg')
train_mean = np.array(train_img).mean()

# 读一张 shufazidian 图
shufa_img = Image.open('/root/sj-tmp/datasets/shufazidian/c/A/0.png')
shufa_mean = np.array(shufa_img).mean()

print(f"训练集平均像素: {train_mean:.1f} (接近0=黑底, 接近255=白底)")
print(f"shufazidian 平均像素: {shufa_mean:.1f}")
# 如果训练集 ~0（黑底），shufazidian ~255（白底）→ 需要 invert
# 如果两者接近 → 不需要 invert
```

### 3.3 探查 3：Original 正楷/草书区分方式（30 分钟）

**目标**：搞清楚 Original 655K 里有没有现成的正楷/草书标签。

需要探查的：
- `CursiveChineseCalligraphyDataset` 目录结构
- Original 数据文件名 pattern（是否含"草书"/"正楷"关键字）
- 是否有 CSV/JSON 元数据文件含书体标签

**决策**：
- 如果有现成标签 → 长尾集可行
- 如果没有 → **放弃长尾集**，只保留主验证集 A（shufazidian）和主验证集 B（端到端）

### 3.4 探查 4：整幅书法图来源（30 分钟）

**目标**：搞清楚主验证集 B（端到端）的 100-200 张整图从哪来。

需要探查的：
- `/caoshu/imgs/` 下现有多少张整图（2.jpg, 6.jpg, 10.jpg...）
- `/caoshu/imgs/samples/` 是否有更多整幅图
- 其他目录（如 `eval/`、`test/`）是否有带标注的整图

**结果**：
- `/caoshu/imgs/`：约 20 张
- `/caoshu/imgs/samples/samples/images/`：**1,000 张整幅书法图**，尺寸 776×2000 ~ 2000×1360，RGB

**决策**：
- ✅ **现有图 1,000+ 张 >> 50 张 → 主验证集 B 来源充足，可直接构建端到端验证集**

### 3.5 探查 5：InternLM 加载可行性（10 分钟）

**目标**：确认 InternLM-7B 能否在现有显存下运行。

**状态**：环境无网络，无法从 HuggingFace 下载。但基于已有证据推断：
- e-IT LoRA 训练（基于 InternLM-7B）此前成功运行 → 模型可加载
- Resampler ~3.2GB + InternLM-7B bf16 ~14GB ≈ 17GB < 24GB

**决策**：
- ✅ **离线路线可行**（Resampler 推理 → JSON → InternLM rerank，不共载）
- ✅ **在线路线理论上可行**（24GB 卡勉强够，但余量小，建议原型阶段先用离线）

---

## 四、P0：建真迹验证集（2-3 天）⭐ P0

### 4.1 核心原则

验证集必须匹配 **Pipeline 真实输入**，不是匹配训练集分布。

### 4.2 ⚠️ 风险：Original 数据集含正楷（重大）

项目记忆里明确写了：Original 655,742 样本是 **"正楷+草书混合"**。

如果直接从 Original 抽样进验证集，大量正楷样本会**稀释"纯草书真迹"信号**，北极星指标被污染。

### 4.3 修正后的验证集设计

| 验证集 | 来源 | 覆盖字类 | 样本数 | 作用 |
|--------|------|----------|--------|------|
| **主验证集 A（字块级）** | shufazidian 验证作者（50 人） | 6,602 字 | ~5K | **最纯净的真迹指标**，0% 正楷污染 |
| **主验证集 B（端到端）** | 真实整幅书法图 → YOLO 切字 | 视图中字数量 | **~1,000 张** | **真正的 Pipeline 端到端指标** |
| ~~辅助长尾集~~ | ~~原 61K 验证集中长尾字的 original~~ | ~~视 Day 0 探查结果~~ | ~~视 Day 0 探查结果~~ | ~~Day 0 确认无现成标签则放弃~~ |

**主验证集 A 只用 shufazidian**，不用 Original 混合。纯净优先。

### 4.4 主验证集 A 的样本量问题

~5K 样本 / 6,602 字 = **平均每字不到 1 张**，指标方差大。

**必须配合置信区间报告**，不能只看单点数字：

```python
import numpy as np
from sklearn.utils import resample

def bootstrap_ci(predictions, labels, n_bootstrap=1000, ci=0.95):
    accs = []
    for _ in range(n_bootstrap):
        idx = resample(range(len(labels)))
        acc = (predictions[idx] == labels[idx]).mean()
        accs.append(acc)
    lower = np.percentile(accs, (1 - ci) / 2 * 100)
    upper = np.percentile(accs, (1 + ci) / 2 * 100)
    return lower, upper

# 输出格式：Top-1: 62.3% (95% CI: 60.1% - 64.5%)
```

### 4.5 避免数据泄漏（关键）

如果 shufazidian 同时用于 P0（验证）和 P1（训练），**必须按作者 hold out**：

```python
import random

all_authors = list(range(493))  # shufazidian 共 493 位作者
random.seed(42)
random.shuffle(all_authors)

train_authors = set(all_authors[:443])   # ~43K 张
val_authors = set(all_authors[443:])     # 50 位作者，~5K 张

# 验证作者的字绝不进训练
# ❌ 不能按样本随机分割
```

### 4.6 Pipeline 一致性陷阱（重大）

shufazidian 是"单字库"（528×538 一张一个字），不是整幅作品。

| 路径 | 做法 | 评估 |
|------|------|------|
| **选择 A（推荐）** | shufazidian 跳过 YOLO，直接过 448 transform | 不完全模拟 pipeline，但避免 YOLO 在单字图上的异常行为 |
| 选择 B | shufazidian 过 YOLO 再切 | YOLO 在单字图上行为不可预测，可能切出异常字块 |

**主验证集 A 用选择 A**（直接 transform）。

**主验证集 B（端到端）必须用选择 B**：从 1,000 张真实整幅书法图中抽样 50-200 张，完整走 `整图 → YOLO → 字块 → Resampler` 流程。这才是真正的 Pipeline 指标。

### 4.7 目录结构

```
/root/sj-tmp/datasets/Validation_Real/
├── main_block/              # 主验证集 A：shufazidian 验证作者，直接 transform
│   ├── 一/
│   ├── 不/
│   └── ...（6,602 字）
├── main_end2end/            # 主验证集 B：整幅书法图 → YOLO → 字块
│   ├── img001/              # 每张整图一个子目录
│   │   ├── 1_天.jpg        # YOLO 切出的字块
│   │   ├── 2_地.jpg
│   │   └── labels.txt      # 人工标注
│   └── ...
└── longtail/                # 辅助长尾集（原 61K 中的草书 original，Day 0 确认可行再做）
    └── ...
```

### 4.8 评估脚本

```python
# 三个指标分开看，不平均
block_top1, block_top5 = evaluate(model, dataloader_main_block)
end2end_top1, end2end_top5 = evaluate(model, dataloader_main_end2end)

print(f"【字块级真迹】Top-1 {block_top1:.2f}% (95% CI: {ci_low:.2f}-{ci_high:.2f})")
print(f"【端到端 Pipeline】Top-1 {end2end_top1:.2f}%")
```

---

## 五、数据集重建方案（v3.0）⭐ 最高优先级

> ⚠️ **数据泄漏确认**：shufazidian 48K 样本在构建 CCC_split 时已混入 Training（resize 至 224×224）。82.15% 验证结果不可信，必须从源头重建。

### 5.1 核心原则

| 类型 | 数据 | 处理方式 |
|------|------|---------|
| **源头真迹** | shufazidian 48K (528×538) | **完整保留原始 PNG**，作为一切处理的黄金源 |
| **源头字体** | Original 655K、V2 300K | **完整保留原始图** |
| **旧合成数据** | CCC_split 中 264K gen | **全部丢弃**，重新生成 |
| **旧预处理** | CCC_split 中 224 原始图 (75K) | **丢弃**，从源头重新处理 |

**原则**：源头是黄金，永不动；合成是可再生的，随时重建。

### 5.2 新目录结构

```
/root/sj-tmp/datasets/CaoshuHQ_v2/
├── source/                          ← 黄金源头，只读
│   ├── shufazidian/                # 528×538 PNG，48K
│   ├── original/                   # Original 655K
│   └── v2/                         # V2 300K
│
├── processed/                       ← 从源头重新预处理
│   ├── train/
│   │   ├── shufazidian_hq/         # 443作者 528→448
│   │   ├── original_hq/            # Original → 448
│   │   └── v2_hq/                  # V2 → 448
│   └── val/
│       ├── shufazidian_block/      # 50作者 528→448（严格隔离）
│       └── ccc_val/                # CCC_split/Val（保留对比）
│
└── synthetic/                       ← 可重新生成的合成数据
    └── gen_v2/                     # 从 448 原始图做 augmentation
```

### 5.3 关键改进：528→448 直接链路

```
旧链路（质量损失）：
shufazidian 528×538 → resize 224×224 → save → train resize 448×448
    （信息损失 82%）              （upsample 模糊）

新链路（质量最优）：
shufazidian 528×538 → resize 448×448 (LANCZOS) → save → train (normalize only)
    （信息保留 85%，一次降采样）
```

**处理脚本**：
```python
from PIL import Image

def process_source_to_hq(src_path, dst_path, target_size=448):
    img = Image.open(src_path).convert('RGB')
    # 直接到目标尺寸，不经过中间 224
    img = img.resize((target_size, target_size), Image.LANCZOS)
    img.save(dst_path, quality=95)
```

### 5.4 验证集隔离（绝对严格）

| 验证集 | 来源 | 处理方式 | 用途 |
|--------|------|---------|------|
| **shufazidian_block** | 50 位验证作者原始 528 | 直接到 448，**绝不进训练** | **北极星指标** |
| CCC_split/Validation | 保留现有 | 作为历史对比基准 | 训练稳定性监控 |

### 5.5 训练数据构成（重建后）

| 数据源 | 占比 | transform | 说明 |
|--------|------|-----------|------|
| shufazidian_hq | 35% | Normalize only | 已经是 448，质量最高 |
| original_hq | 35% | Normalize only | 已经是 448 |
| v2_hq | 15% | Normalize only | 已经是 448 |
| synthetic/gen_v2 | 15% | augmentation | 从 448 生成，可任意数量 |

---

## 六、P1：基于重建数据集的增量训练（5-7 天）

### 6.1 策略

基于 `CaoshuHQ_v2/processed/` 启动全新训练（或从 best.pt 恢复微调）。

### 6.2 预处理（Day 0 已确认）

不需要 invert（shufazidian 与训练集均为黑底）。

### 6.3 作者分层采样（修正公式）

❌ 错误：`weights = [1.0 / author_freq[aid]]` → 完全抹平作者频率，高产作者丰富样本被浪费。

✅ 正确：平方根或对数软平衡，保留高产作者的数据价值，同时让小作者被见到。

```python
import math
from torch.utils.data import WeightedRandomSampler

# 方案 1：平方根平衡（推荐）
weights = [1.0 / math.sqrt(author_freq[aid]) for aid in author_ids]

# 方案 2：对数平衡（更柔和）
# weights = [1.0 / math.log(author_freq[aid] + 1) for aid in author_ids]

sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
```

### 6.4 训练作者 / 验证作者划分

同 P0 的 hold out：443 位训练 / 50 位验证，验证作者绝不进训练。

### 6.5 早停策略（不固定步数）

❌ 不要："训练到 120K 步"

✅ 要："在真迹验证集上 Top-1 连续 N 次评估不再上升 → 早停"

```python
# 每 eval_interval 步评估一次真迹验证集
best_val_top1 = 0
patience = 5  # 连续 5 次不提升就停
no_improve = 0

for step in range(current_step, max_step):
    train(...)
    if step % eval_interval == 0:
        val_top1 = evaluate(model, val_real_dataloader)
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            save_checkpoint(f"best_val_top1_step{step}.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at step {step}, best val Top-1: {best_val_top1:.2f}%")
                break
```

---

## 七、P2b：LLM 纯文本 rerank（1-2 天）⭐ 最快见效

### 6.1 核心洞察

Top-5 74.91% → Top-1 53.83%，**21% 的鸿沟 = 正确答案在候选里，只是排序错了**。

### 6.2 路线选择

| 路线 | 说明 | 评估 |
|------|------|------|
| ❌ e-IT LoRA rerank | e-IT LoRA 设计目的是**看图生成文本**（生成路线），不是 rerank scorer | 已证伪的端到端路线，不要复用 |
| ✅ **纯文本 rerank（推荐）** | 给原始 InternLM 看上下文 + Top-5 候选，让它选最合理的 | 纯文本任务，不需要 LoRA，不需要看图 |

### 6.3 Prompt 设计

```python
prompt = f"""这是一幅书法作品的 OCR 识别结果。以下识别可能有错，请谨慎判断。

已识别的前文（可能包含错误）：{previous_chars}

当前位置的字，视觉模型给出的候选（按置信度从高到低）：
1. {candidate[0]}
2. {candidate[1]}
3. {candidate[2]}
4. {candidate[3]}
5. {candidate[4]}

请根据上下文语义和常见书法内容（古诗、经典文本），判断当前位置最合理的字是哪一个。只输出字本身，不要解释。

答案："""
```

**关键细节**：明确标注"前文可能包含错误"，避免 LLM 被错误上下文带偏。

### 6.4 触发阈值：用校准曲线决定（不是拍脑袋 40%）

❌ 不要：`if top1_confidence < 40%: rerank()`

✅ 要：先做**置信度 vs 正确率的校准曲线**，找到"置信度虚高"的拐点。

```python
# 在验证集上分桶统计
buckets = {
    '0-20%':  {'total': 0, 'correct': 0},
    '20-40%': {'total': 0, 'correct': 0},
    '40-60%': {'total': 0, 'correct': 0},
    '60-80%': {'total': 0, 'correct': 0},
    '80-100%': {'total': 0, 'correct': 0},
}

for img, label in val_loader:
    top1_conf, top1_pred, top5_candidates = model(img)
    bucket = get_bucket(top1_conf)
    buckets[bucket]['total'] += 1
    if top1_pred == label:
        buckets[bucket]['correct'] += 1

# 找到正确率明显低于置信度的"虚高"拐点
for bucket, stats in buckets.items():
    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    print(f"{bucket}: 正确率 {acc:.2%} (n={stats['total']})")
```

**rerank 触发阈值 = 校准曲线上的拐点**（可能是 30%，也可能是 60%，看数据说话）。

### 6.5 上下文策略

前面 N 字也是模型预测的，可能是错的。**不要给 LLM 盲信**。

可选策略：
1. **标注警告**：如 prompt 中所示，明确写"以下识别可能有错"
2. **缩短上下文**：N=5-10 字，不要 N=20（错误累积更多）
3. 多候选前文暂不实现（原型阶段保持简单）

### 6.6 实现路线：先离线，后在线

| 路线 | 做法 | 适用阶段 |
|------|------|----------|
| **离线 rerank（推荐原型阶段）** | CaoshuReader 跑完整图 → 存 JSON（位置+Top-5+上下文）→ 另一脚本用 InternLM 逐条 rerank → 合并结果 | 简单，不卡显存，快速验证"LLM 是否能 rerank" |
| 在线 rerank | pipeline 里同时加载 Resampler + InternLM | 显存可能吃紧（3.2GB + ~14GB bf16 ≈ 17GB+），确认有效后再搞 |

**Day 5-7 原型阶段先走离线路线**。如果 rerank 有效，再考虑在线集成。

### 6.7 预期效果

| 场景 | 现有 Pipeline | + LLM rerank |
|------|--------------|--------------|
| 高置信度（>阈值）| 直接输出 Top-1 | 不变 |
| 低置信度（<阈值）| 输出 Top-1（可能错）| LLM 从 Top-5 重排，选语义最合理的 |

保守估计：Top-1 从 53.83% → **60%-68%**（填平部分 21% 鸿沟）。

---

## 八、P2a：混淆矩阵分析（1-2 天）

### 7.1 目标

用真迹验证集跑一遍，输出系统性的错误模式报告。

### 7.2 输出清单

```python
# 1. 字类准确率排行（最差 100 字）
# 2. Top-5 混淆对热力图
# 3. 混淆模式分类：
#    - 形近字（如 倨/瀣/倔）
#    - 同作者跨字（风格耦合）
#    - 低频字被高频字吞噬
```

### 7.3 指导后续动作

| 混淆模式 | 根因 | 对策 |
|----------|------|------|
| 形近字 | 局部特征区分度不足 | 需要更细粒度特征 / LLM rerank |
| 同作者跨字 | 作者风格信息耦合到字 embedding | shufazidian 作者分层训练解耦 |
| 低频被高频吞噬 | 长尾字 embedding 学习不充分 | shufazidian 补充后可能自愈 |

---

## 九、P3：Loss 反弹调查（归档，不阻塞）

### 8.1 现状

- step 91877: loss 0.1400（best）
- step 100000: loss 0.4496（3 倍反弹）

### 8.2 决策：不深查

| 理由 | 结论 |
|------|------|
| best checkpoint（step 91877）已落地使用 | 不影响当前推理 |
| 除非要继续训练或复现实验 | 当前不需要 |
| 排查耗时 1-2 天，ROI 低 | **归档为后续排查项** |

---

## 十、跳过项

### ❌ 不做的

| 项目 | 原因 |
|------|------|
| 验证集对齐到训练分布（27% gen） | 方向错了，指标变水，对 Pipeline 没帮助 |
| 用 e-IT LoRA 做 rerank | LoRA 是生成路线，不是 rerank scorer |
| P3 深入调查 | best checkpoint 已落地，不阻塞主线 |
| Original 直接进验证集不筛草书 | 正楷会污染纯草书真迹信号 |
| YOLO 切 shufazidian 单字图 | 行为不可预测，和整图切字路径不一致 |

---

## 十一、修正后的执行路线图

```
Day 0（今天，1-1.5 小时）
├── 探查 1：前置实验 + 配对对照（15 分钟）
│   └── 输出：shufazidian 48K Top-1 vs CCC/Val original Top-1
├── 探查 2：颜色空间 sanity check（5 分钟）
│   └── 输出：是否需要 invert
├── 探查 3：Original 正楷/草书区分方式（30 分钟）
│   └── 输出：长尾集是否可行
├── 探查 4：整幅书法图来源（30 分钟）
│   └── 输出：端到端验证集规模
└── 探查 5：InternLM 加载显存测试（10 分钟）
    └── 输出：离线/在线路线决策

Week 1
├── Day 1:   P0 —— 划分 shufazidian 训练/验证作者（443+50）
├── Day 2:   P0 —— 建主验证集 A（shufazidian 验证作者，~5K，直接 transform）
├── Day 3:   P0 —— 建主验证集 B（整幅书法图 → YOLO 切字）
│            P0 —— 跑现有模型，拿三个指标 + 置信区间
├── Day 4-7: P2b —— LLM 纯文本 rerank 原型（离线路线）
│            └── 校准曲线 + 阈值确定 + 前后对比

Week 2
├── Day 1-2: P1 —— shufazidian 预处理（以 Day 0 颜色检查为准）+ 接入训练
├── Day 3:   P1 —— 训练启动（B1 追加 + 作者分层采样 sqrt）
├── Day 4-7: P1 —— 训练（早停：真迹验证集 Top-1 连续 5 次不升则停）

Week 3
├── Day 1-2: P1 —— 训练完成，用真迹验证集 A/B 分别评估
│            └── 输出：新模型 vs 旧模型对比
├── Day 3-4: P2a —— 混淆矩阵分析（用真迹验证集）
│            └── 输出：最差 100 字 + Top 混淆对 + 模式分类
└── Day 5:   复盘 —— 决定下一步（长尾增强 / rerank 精修 / 继续训练）
```

---

## 十二、快速决策

| 你想先看到什么 | 做哪个 | 周期 |
|---------------|--------|------|
| **消掉最大不确定性** | Day 0 五个探查 | 1-1.5 小时 |
| **真实基线** | P0 建真迹验证集 | 3-4 天 |
| **最快提升 Pipeline 准确率** | P2b LLM rerank | 2-3 天 |
| **系统提升模型本身** | P0 → P1 | 2-3 周 |
| **全部一起做** | Track 1 + Track 2 并行 | 2-3 周 |

---

*文档版本：v2.3 最终版*  
*修正历史：v1.0 方向错误 → v2.0 对齐真实场景 → v2.1 落地细节 → v2.2 风险修正 → v2.3 Day 0 探查前置（配对对照 / 颜色检查 / 草书区分 / 整图来源 / 显存测试）*
