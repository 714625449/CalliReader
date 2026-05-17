# CaoshuReader HQ 训练总结与 Contrastive Loss 行动计划

> 日期：2026-05-16  
> 状态：100K steps 基础训练已完成，Contrastive Loss 修改就绪，等待启动

---

## 一、训练概况

### 1.1 配置

| 项目 | 配置 |
|------|------|
| **数据集** | HQ（SFZD 46,420 + CCC 8,580 = 56,650 张图） |
| **数据划分** | Train: 41,954 / Val: 5,374 / Test: 4,659 |
| **字符集** | 5,374 类（val 中至少出现 1 次的字） |
| **在线增强** | Rotation ±8°, Perspective, ColorJitter, ElasticTransform |
| **采样倍率** | Train ×3 |
| **模型** | PerceiverResampler（0.571B，唯一可训练模块） |
| **Batch Size** | 16 × grad_accum 16 = 256 |
| **LR** | 1e-4，CosineAnnealingWarmRestarts(T_0=5000 scheduler steps) |
| **Loss** | `alignment_loss` = mean(1 - cosine_similarity) |
| **总步数** | 100,000 |
| **耗时** | ~5 天（单卡 RTX 3090） |

### 1.2 核心指标

| 指标 | 数值 | Step |
|------|------|------|
| **最终 Train Loss** | 0.0566 | 100,000 |
| **最佳 Train Loss** | **0.0108** | 69,716 |
| **最终 Val Top-1** | 36.53% | 100,000 |
| **最佳 Val Top-1** | **36.58%** | 99,000 |
| **最佳 Val Top-5** | **53.59%** | 90,000 |

---

## 二、阶段拆解分析

### Phase 1：快速上升期（0 ~ 20K）

```
Val Top-1:   0.02% → 22.96%  (+22.94%)
Val Top-5:   0.20% → 39.41%  (+39.21%)
效率:        ~1.1% / 1K steps
```

模型从零开始学习视觉-文本对齐，每 1K steps 涨 1-2%，这是训练最有效率的阶段。

### Phase 2：缓慢增长期（20K ~ 40K）

```
Val Top-1:  22.96% → 34.07%  (+11.11%)
Val Top-5:  39.41% → 51.64%  (+12.23%)
效率:       ~0.55% / 1K steps
```

增长速度减半，30K 之后每 1K steps 仅提升 0.3-0.5%。

### Phase 3：完全停滞期（40K ~ 80K）⚠️

```
Val Top-1:  34.07% → 34.80%  (+0.73%  整整 40K steps!)
Val Top-5:  51.64% → 51.86%  (+0.22%)
效率:       ~0.018% / 1K steps
```

**这是最严重的问题**：  
- Train loss 从 ~0.05 持续优化到 **0.0108**（优化了 78%）
- Val Top-1 在 **34.7% 附近横盘了 4 万步**
- 典型的 **surrogate loss 与下游 metric 脱节**

### Phase 4：LR Restart 震荡恢复期（80K ~ 100K）

```
Val Top-1:  34.80% → 36.53%  (+1.73%)
Val Top-5:  51.86% → 53.03%  (+1.17%)
效率:       ~0.087% / 1K steps
```

- 80K restart 后短期骤降至 **33.62%**（step 81K）
- 之后缓慢爬升，最终达到 36.58%
- Restart 效率是停滞期的 **4.8 倍**，但绝对收益仍有限

---

## 三、核心诊断

### 诊断 1：alignment_loss 的天花板

`alignment_loss` 只优化一个方向：**拉近视觉特征与对应文本 embedding**。它不优化另一个关键方向：**推远视觉特征与其他相似字的 embedding**。

后果：
- Train loss 可以无限低（方向一致即可）
- 但 retrieval 时，形近字（"茶"vs"荼"、"己"vs"已"）的视觉特征和文本 embedding 都聚集在相近区域
- 模型无法区分它们 → Val Top-1 卡住

### 诊断 2：不是传统过拟合

这不是"背答案"式的过拟合：
- 数据集严格隔离了 50 个 val 作者，无泄漏
- Train loss 低是因为 loss function 本身的局限，不是 memorization
- 证据：Top-5（53%）远高于 Top-1（36%），说明正确答案常在候选列表里，只是排不到第一

### 诊断 3：LR Restart 的再评估

| 阶段 | 步数 | ΔTop-1 | 效率 |
|------|------|--------|------|
| 40K-80K（衰减） | 40K | +0.73% | 0.018%/K step |
| 80K-100K（restart） | 20K | +1.73% | 0.087%/K step |

Restart 的效率确实比继续衰减高 4.8 倍。问题不是 restart 策略不好，而是 **alignment_loss 本身在 34% 附近遇到了结构性天花板**——不管 lr 怎么调，没有负样本 pushing apart 的机制，模型就是分不开形近字。

---

## 四、下一步行动计划

### 4.1 目标

在保持现有 retrieval 范式不变的前提下，通过 **Contrastive Loss + Hard Negative Mining** 突破 36.5% 天花板。

### 4.2 为什么选 Contrastive Loss（优先级 #1）

| 方案 | 优点 | 缺点 | 优先级 |
|------|------|------|--------|
| **Contrastive Loss** | 直接解决类间不可分，改动最小（只动 loss） | batch 内 hard negative 出现频率低 | **#1** |
| 分类头 | 简单有效 | 改变评估范式（retrieval → classification），部署逻辑要改 | #2 |
| 部首结构先验 | 草书场景高价值 | 需要额外数据和预处理 | #3 |
| 解冻 text embedding | 潜在收益大 | 风险最高，容易破坏预训练空间结构 | #4 |

### 4.3 具体修改

已在 `caoshu/train.py` 中实现 `contrastive_alignment_loss`：

```python
def contrastive_alignment_loss(pred, target_embed, labels, margin=0.1, neg_weight=0.5):
    # 1. 原 alignment loss（拉近正样本）
    tgt = F.normalize(target_embed, dim=-1).unsqueeze(1)
    pred_norm = F.normalize(pred, dim=-1)
    cosine_sim = (pred_norm * tgt).sum(dim=-1)
    pos_loss = (1 - cosine_sim).mean()

    # 2. Contrastive push-apart（推开 batch 内负样本）
    pred_vec = F.normalize(pred.mean(dim=1), dim=-1)    # [B, D]
    target_norm = F.normalize(target_embed, dim=-1)      # [B, D]

    # batch 内负样本 mask（同 label 不算负样本）
    neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()

    # 视觉特征 vs 文本 embedding 的相似度矩阵
    sim_matrix = torch.matmul(pred_vec, target_norm.t())  # [B, B]

    # hard negative: 只惩罚相似度 > margin 的负样本对
    neg_loss = F.relu(sim_matrix - margin) * neg_mask
    neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-8)

    return pos_loss + neg_weight * neg_loss
```

关键参数：
- `margin=0.1`：只惩罚相似度 > 0.1 的负样本对（高维空间中随机向量期望正交 ≈ 0）
- `neg_weight=0.5`：负样本项系数，避免主导训练
- Batch 内负样本比例 ≈ 100%（6766 类、batch=16，同 batch 重复概率 ~0.02%）

### 4.4 启动命令

```bash
cd /caoshu/caoshu
setsid bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate caoshu && python3 -u train.py --skip_disk_check --resume /root/sj-tmp/checkpoints/CaoshuReader_HQ/caoshu_step99000.pt --total_steps 120000 > /root/sj-tmp/checkpoints/CaoshuReader_HQ/train_contrastive.log 2>&1"
```

配置说明：
- **Resume from**: `caoshu_step99000.pt`（Val Top-1 最高：36.58%）
- **Total steps**: 120,000（再跑 21K steps）
- **LR schedule**: 保持原 CosineAnnealingWarmRestarts 不变，观察模型能否适应新 loss
- **Log**: `train_contrastive.log`（新文件，与基础训练区分）

### 4.5 预期与观察指标

| 观察项 | 预期 |
|--------|------|
| Train loss | 初期可能上升（加了负样本项），然后重新下降 |
| Val Top-1 | 1K-3K steps 内应能看到是否突破 36.5% |
| 震荡情况 | 如果 val 剧烈震荡，降低 neg_weight 或考虑 lr 重启 |
| Hard negative 效果 | 第一轮用随机 batch，如果效果不足，后续考虑 class-aware sampling（刻意塞入形近字对） |

---

## 五、Checkpoint 文件索引

```
/root/sj-tmp/checkpoints/CaoshuReader_HQ/
├── caoshu_best.pt          → step 69716, loss 0.0108（train loss 最佳）
├── caoshu_final.pt         → step 100000, loss 0.0566（最终模型）
├── caoshu_step99000.pt     → step 99000, Val Top-1 36.58%（Val 最佳）⭐
├── caoshu_step86000.pt
├── caoshu_step87000.pt
├── caoshu_step88000.pt
├── caoshu_step89000.pt
├── caoshu_step90000.pt
├── caoshu_step91000.pt
├── caoshu_step92000.pt
├── caoshu_step93000.pt
├── caoshu_step94000.pt
├── caoshu_step95000.pt
├── caoshu_step96000.pt
├── caoshu_step97000.pt
├── caoshu_step98000.pt
└── caoshu_step100000.pt
```

---

## 六、历史对比

| 模型 | 数据集 | 训练样本 | Val Top-1 | 备注 |
|------|--------|---------|-----------|------|
| callialign.pth（旧） | CCC | 6,493 | 29% | 小数据集，欠拟合 |
| 合并数据集版 | CCC+SFZD | 974K | 47.79% | 有 test 泄漏（数据问题） |
| **HQ 版（当前）** | **HQ（严格隔离）** | **126K（×3 增强）** | **36.58%** | **无泄漏，真实能力** |

> 注：合并数据集版的 47.79% 因 test 泄漏不可信。HQ 版的 36.58% 是严格隔离作者后的真实指标。

---

---

## 七、关键发现：Token 碰撞问题（2026-05-16 下午）

### 7.1 发现

离线检查 tokenizer 映射时发现：**17.27% 的字符（928 / 5374 字）共享同一个 token ID**。

| Token ID | 共享字数 | 占比 |
|---------|---------|------|
| 236 | **237 字** | 4.41% |
| 235 | **197 字** | 3.67% |
| 234 | **185 字** | 3.44% |
| 233 | **158 字** | 2.94% |
| 232 | **144 字** | 2.68% |
| 231 | **7 字** | 0.13% |

这意味着这 928 个字的目标 embedding **完全相同**，模型在 retrieval 时无法通过 embedding 区分它们。

### 7.2 影响

- 这些字的 `alignment_loss` 目标完全一致 → 视觉特征被强行压缩到同一点
- Contrastive loss 对它们**完全无效**（目标 embedding 相同，推到哪里都是同一个点）
- 这是 Top-1 卡在 36% 的**结构性原因之一**

### 7.3 解决方案：Learnable Character Embedding

用 `nn.Embedding(num_classes, 4096)` 替代 `tok_embeddings` 查表：

```python
# 原做法（有碰撞）
tgt_embed = tok_embeddings(token_ids)

# 新做法（每个字独立）
char_embeddings = nn.Embedding(num_classes, 4096)
tgt_embed = char_embeddings(labels)
```

**初始化策略**：
- 有独立 token 的字：用 `tok_embeddings(token_id)` 初始化，保留预训练语义结构
- 碰撞字：同样初始化（初始值相同），但训练后自然分离

**优点**：
- 每个字都有唯一的 target embedding，彻底解决碰撞
- 不需要换 tokenizer、不需要重新处理数据
- 工程量：~20 行代码
- 可与 resampler 联合训练，embedding 空间自适应

### 7.4 执行顺序（已确定）

```
Step 1: Contrastive loss（当前）
        ↓ 跑 10-20K 步，看独立 token 的字能涨多少
Step 2: 如果碰到天花板（~40%，碰撞字的上限隐约可见）
        ↓ 加 learnable char_embedding（~20 行）
Step 3: 如果还不够
        ↓ 考虑换 tokenizer（最后手段）
```

---

## 八、代码改动记录

### 已修改文件

| 文件 | 改动内容 |
|------|---------|
| `caoshu/train.py` | 1. 新增 `contrastive_alignment_loss()`（pos_loss + neg_loss）<br>2. 训练循环传入 `labels`<br>3. `labels = labels.to(device)` |

### 待修改（等待启动后视效果决定）

| 文件 | 改动内容 | 触发条件 |
|------|---------|---------|
| `caoshu/train.py` | 1. 新增 `nn.Embedding(num_classes, 4096)`<br>2. Optimizer 加入 char_embeddings 参数<br>3. 验证时用 char_embeddings 生成 all_char_embeds | Step 1 效果不足 |

---

*更新时间：2026-05-16*  
*状态：Contrastive Loss 代码已改，Token 碰撞问题已发现，Learnable Embedding 方案已确定，等待启动指令*
