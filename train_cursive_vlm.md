# 纯草书数据微调 InternVL 详细计划

## 一、当前问题诊断

| 指标 | 原始 CalliReader | 当前 CaoshuReader (Resampler-only) |
|------|----------------|----------------------------------|
| 整页 OCR F1 | **0.885** | 0.516 |
| 单字 Top-1 | ~90% | 0~35% |
| NED | **0.131** | 0.947 |

**根因**：当前只训练 Resampler（0.57B），LLM（7B）完全冻结。Resampler 输出发生 collapse，LLM 也无法利用语言先验纠错。

**目标**：在原始 CalliReader 基础上，用纯草书数据继续微调，提升草书识别能力。

---

## 二、总体策略：分阶段 e-IT + LoRA

CalliReader 论文使用 **e-IT（embedding-aligned Instruction Tuning）**：
1. Resampler 提取单字视觉特征 → `reference_embeds`
2. 通过 `[UNUSED_TOKEN_140]` 注入 LLM 输入层
3. LLM 根据这些 embedding 生成对应字符

我们将分 4 个阶段执行：

```
阶段0: 重新训练 Resampler（纯草书，防 collapse）
阶段1: 预计算所有单字 embedding
阶段2: 构建 e-IT 训练数据
阶段3: 用 xtuner LoRA 微调 LLM
阶段4: 验证整页 OCR
```

---

## 三、阶段详解

### 阶段0：重新训练 Resampler（纯草书）

**问题**：`callialign.pth` 是混合数据（正楷+草书）训练的，在纯草书上表现一般。当前从头训练的 checkpoint 发生 collapse。

**方案**：
- 加载 `callialign.pth` 作为初始化（而非从头）
- 在 `CCC_split/Training` 上继续训练
- **关键修改**：降低学习率，加入 early stopping，监控 Validation 准确率而非 loss
- 训练目标：Validation Top-1 达到 50%+（目前 35%）

**训练脚本**：`caoshu/train.py` 需修改：
```python
# 加载 callialign.pth 作为预训练权重
ckpt = torch.load('/caoshu/params/callialign.pth')
resampler.load_state_dict(ckpt['model_state_dict'])

# 学习率降至 1e-5
lr = 1e-5

# 每 1000 步评估一次 Validation 准确率
if step % 1000 == 0:
    evaluate_topk(...)
```

**预期**：2~4 小时，Validation 准确率从 35% 提升到 50~60%。

---

### 阶段1：预计算所有单字 Embedding

用阶段0优化后的 Resampler，对 `CCC_split/Training` 所有 338,870 张图片预计算 embedding。

**脚本逻辑**：
```python
for img_path, label in dataset:
    img_tensor = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        vit_feats = get_visual_embed(img_tensor, vit, mlp1)
        embed = resampler(vit_feats).mean(dim=1)  # (1, 4096)
    torch.save(embed.cpu(), f'/root/sj-tmp/datasets/CCC_embeddings/{idx}.pt')
```

**输出**：每个单字一个 `.pt` 文件，总大小约 5.5 GB（338k × 4096 × 2 bytes）。

---

### 阶段2：构建 e-IT 训练数据

将单字数据转换为 LLaVA SFT + embedding 格式。

**JSON 格式**（每条记录）：
```json
{
  "id": 0,
  "image": "Training/哀1/xxx.jpg",
  "embedding": "/root/sj-tmp/datasets/CCC_embeddings/0.pt",
  "conversations": [
    {"from": "human", "value": "<image>\n读一下这个字。[UNUSED_TOKEN_140]"},
    {"from": "gpt", "value": "哀"}
  ]
}
```

**关键**：
- 每个样本只有一个 `[UNUSED_TOKEN_140]`（对应一个单字 embedding）
- `image` 指向原始单字图片（用于 ViT 特征提取，e-IT 中可省略，但保留以便端到端微调）
- `embedding` 指向阶段1预计算的 `.pt` 文件

**数据量**：Training 338,870 条，Validation 61,181 条。

---

### 阶段3：用 xtuner LoRA 微调 LLM

**配置**：基于 `/caoshu/train/xtuner/configs/internvl/v2/internvl_v2_internlm2_5_8b_lora_finetune.py` 修改。

**关键修改**：

```python
# Model
path = '/caoshu/InternVL'  # 本地已修改的 InternVL（含 CalliReader 代码）

# Data
data_path = '/root/sj-tmp/datasets/CCC_cursive_eit.json'
image_folder = '/root/sj-tmp/datasets/CCC_split'

# 冻结 ViT，微调 Resampler + LLM LoRA
model = dict(
    type=InternVL_V1_5,
    model_path=path,
    freeze_llm=True,        # LLM 主体冻结，LoRA 可训练
    freeze_visual_encoder=True,
    llm_lora=dict(
        type=LoraConfig,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=None,
        task_type='CAUSAL_LM'),
)

# 训练参数
batch_size = 16
accumulative_counts = 4
lr = 1e-5
max_epochs = 1
```

**训练命令**：
```bash
cd /caoshu/train
NPROC_PER_NODE=1 xtuner train \
    xtuner/configs/internvl/v2/cursive_eit_lora.py \
    --work-dir /root/sj-tmp/checkpoints/CursiveEIT
```

**显存预估**：QLoRA 约 16~20 GB，LoRA 约 20~23 GB。

**训练目标**：让 LLM 学会将 `[UNUSED_TOKEN_140]` 位置的 embedding 映射到正确的字符 token。本质上是一个从视觉 embedding 到字符的"翻译"任务。

---

### 阶段4：验证整页 OCR

训练完成后：
1. 用 `convert_to_official.py` 转换权重
2. 在 `samples` 的 20 张图上运行 `test_samples_callireader.py`
3. 对比训练前后的 F1/NED

**成功标准**：
- 如果原始 CalliReader 在草书上 F1 约为 0.7~0.8（从 20 张样本推测）
- 微调后目标：F1 > 0.85，NED < 0.15

---

## 四、为什么这比只训 Resampler 更好？

| 维度 | 只训 Resampler | e-IT 微调 LLM |
|------|--------------|--------------|
| 训练参数 | 0.57B | 7B (LoRA 约 0.5B) |
| 损失函数 | 余弦相似度 | CrossEntropy（字符生成） |
| LLM 参与 | ❌ 完全冻结 | ✅ 生成字符 |
| 语言先验 | ❌ 无 | ✅ 利用 LLM 先验纠错 |
| 训练数据 | 单字图片 | 单字图片 + 字符文本 |
| 端到端能力 | ❌ 需 YOLO+排序 | ✅ 直接生成 |

---

## 五、风险与备选方案

| 风险 | 应对措施 |
|------|---------|
| 24GB 显存不足 | 使用 QLoRA（`quantization_llm=True`） |
| Resampler 本身不够好 | 阶段0先优化 Resampler 到 50%+ |
| 单字数据无法提升整页 OCR | 可尝试将多个单字拼接成"伪整图"训练 |
| xtuner 与 CalliReader 代码不兼容 | 已验证 `modeling_internvl_chat.py` 的 `forward` 和 `generate_ocr` 与 xtuner 兼容 |

---

## 六、下一步行动

1. ✅ **立即可做**：阶段0——修改 `train.py`，从 `callialign.pth` 加载，降低 lr 继续训练 Resampler
2. **随后**：阶段1——预计算 embedding（约 2 小时）
3. **随后**：阶段2——生成 e-IT JSON（约 30 分钟）
4. **随后**：阶段3——xtuner LoRA 训练（约 6~12 小时）
5. **最后**：阶段4——验证效果

需要我立即开始实施阶段0（修改训练脚本）吗？
