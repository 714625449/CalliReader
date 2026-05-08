# DeepSeek Prompt：CursiveCalliReaderTrainer 训练策略咨询

请将以下完整 Prompt 复制到 DeepSeek 对话中：

---

```text
你是一位专注于书法 OCR 和视觉-语言模型微调的专家，角色名为 CursiveCalliReaderTrainer。请基于我提供的详细实验背景，给出针对性的 Resampler 训练策略优化建议。

## 1. 系统架构

项目名称：CalliReader（基于 InternVL2.5-8B 的书法 OCR）
Pipeline：
  整页书法图像 → YOLO 检测 → ViT 特征提取 → Perceiver Resampler → VQ 量化 → InternLM2-8B LLM 解码

关键组件：
- YOLO：检测单字/文本行位置
- ViT：编码视觉特征
- Perceiver Resampler：将 ViT patch tokens 压缩为固定数量 learnable query tokens
- VQ（Vector Quantization）：将 Resampler 输出离散化为视觉 token
- LLM（InternLM2-8B）：根据视觉 token 生成 OCR 文本

原始超参数：num_layers=4, num_learns=3, num_image_token=3（ViT patch 数）

## 2. 训练实验详情

### 训练数据
- CaoshuCC-20k：约 20,000 张草书单字图像
- 训练目标：让 Resampler 将草书视觉特征更好地对齐到 LLM 语义空间

### 训练配置
- Resampler：num_layers=8, num_learns=12, dropout=0.1
- 总步数：100,000 steps
- 最佳验证点：step 52,000，val_loss=0.8391（caoshu_best_val_1.pt）
- 最终点：step 100,000（caoshu_final.pt）
- 训练损失在 ~30k steps 后趋于平稳

### 评估结果

A) 单字分类（CCC Test Set, 200 samples）
- Original CalliReader：10.5% 准确率
- CaoshuReader：7.0% 准确率（低于基线）
- 注意：单字测试被认为不能反映整页 OCR 能力，因为系统依赖 YOLO+LLM 的联合 pipeline

B) 整页 OCR（15 张测试图，含真书/行书/草书）
- Original：平均 CER 81.46%，15/15 成功
- CaoshuReader（best_val）：
  - 原始平均 CER 875.56%（6 张 OOM 被惩罚为 999%+）
  - 共同成功样本（9 张）加权 CER：93.78%（vs Original 99.52%）
  - 相对改善：5.7%

### 逐样本亮点
- 13.jpg（密集草书）：Caoshu CER 82.9% vs Original 109.8%（↓26.8%）
- 2_3.jpg / 2_4.jpg：Caoshu 150% vs Original 160%（↓10%）
- 20.jpg：Caoshu 开头正确识别"为问西风几时来"，但后续 LLM 幻觉严重

## 3. 已诊断的关键问题

### 问题 1：LLM 幻觉（最严重）
- 7.jpg：GT 是"素心愛雲水..."，但两个模型都输出"春色满园关不住一枝红杏出墙来"
- 20.jpg：Caoshu 正确识别开头后，LLM 续写完全不同的古诗
- 根因：InternLM2-8B 的强语言先验，未经书法 OCR 语料微调

### 问题 2：OOM（CaoshuReader）
- 8层/12query Resampler 在 RTX 3090 24GB 上，6/15 张图 OOM
- 原模型 4层/3query 全部成功
- OOM 修复（expandable_segments）对 Caoshu 效果有限

### 问题 3：训练目标与推理任务不匹配
- 训练：单字草书图像 → Resampler 对齐
- 推理：整页图像 → YOLO 检测 → 多区域聚合 → LLM 生成
- Gap：训练未覆盖 YOLO、多区域聚合、LLM 文本生成等环节

### 问题 4：重复生成
- 部分输出循环重复（如"汉兵一百八十八..."反复出现）
- 当前 repetition_penalty=1.0

## 4. 我的问题

请针对以上背景，回答以下问题：

Q1: **Resampler 训练策略优化**
   当前训练目标（单字草书 → Resampler 特征对齐）是否与整页 OCR 推理存在根本性 mismatch？
   如果要继续优化 Resampler，应该：
   a) 保持单字训练，但改用对比学习（CLIP-style）或知识蒸馏？
   b) 改为整页级别的训练（需要 1000+ 整页图文对）？
   c) 在 Resampler 后接一个小型解码器做中间监督（如单字识别头）？
   d) 其他你认为更好的方案？

Q2: **LLM 微调策略**
    hallucination 是主要瓶颈。在显存受限（24GB）的情况下，推荐哪种 LLM 微调方案？
   a) LoRA（rank 多少合适？target_modules？）
   b) QLoRA（4-bit 量化 + LoRA）
   c) Prompt tuning / Prefix tuning
   d) 全量微调（需要更大显存，可能不可行）
   e) 两阶段：先 LoRA 微调 LLM，再联合微调 Resampler+LLM？

Q3: **Resampler 架构调整**
   8层/12query 导致 OOM，但 4层/3query 性能不足。有没有中间方案？
   例如：6层/8query、动态 query 数量、分层 Resampler（先粗后细）、或者把 Resampler 换成更轻量的架构（如 Transformer Pooling）？

Q4: **数据策略**
   目前只有 20k 单字草书数据，缺乏整页训练数据。在数据有限的情况下：
   a) 如何有效利用现有单字数据（数据增强、合成整页）？
   b) 需要多少整页图文对才能看到 LLM 微调的明显效果？
   c) 是否有开源书法 OCR 数据集推荐？

Q5: **训练超参数调优**
   训练损失在 30k steps 后 plateau，但继续训练到 100k 才得到 best_val @ 52k。
   这是否说明：
   a) 学习率过高（需要 cosine decay / warmup 调整）？
   b) batch size 太小（当前可能是 1~2）？
   c) 需要梯度累积或混合精度优化？
   d) 早停策略应该如何设置（patience=?, delta=?）？

Q6: **评估与验证**
   当前用 CER 评估整页 OCR，但 LLM 幻觉导致 CER 很高（80~100%）。
   有没有更好的评估指标或方法？
   例如：BLEU/ROUGE 是否比 CER 更适合生成式 OCR？
   或者应该把评估拆分为：检测准确率 + Resampler 特征相似度 + LLM 生成质量？

请给出结构化、可落地的建议，并说明每个建议的预期收益和实施成本。
```

---

## 使用说明

1. 打开 [DeepSeek Chat](https://chat.deepseek.com/) 或任何支持长上下文的 LLM
2. 将上方 ` ```text ` 到 ` ``` ` 之间的内容完整复制粘贴
3. 等待模型回复（通常需要 1~2 分钟的长思考）
4. 根据回复中的建议，制定下一阶段的实验计划
