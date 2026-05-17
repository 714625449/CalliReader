# CaoshuReader 数据集重建方案 v4.0（2026-05-15）

> 核心原则：shufazidian 真迹为主，CCC 原图补充，图形质量最优先  
> 前提：CCC_split 与 CaoshuMerged 已删除，旧数据清空

---

## 一、数据源精确定义

### 1.1 CCC 数据集（原 CursiveChineseCalligraphyDataset）

| 属性 | 数值 |
|------|------|
| 字体类型 | 各种书法字体（非纯草书） |
| 图片尺寸 | **96×96** |
| 总字数 | 5,301 个字 |
| 总图数 | 673,164 张 |

**子集分布**：

| 子集 | 数量 | 性质 | 覆盖字类 |
|------|------|------|---------|
| 训练集 | 655,892 | 原图 25,557 + **合成 630,185** | 6,493 |
| 验证集 | 7,724 | **全原图，无合成** | 4,281 |
| 测试集 | 9,548 | **全原图，无合成** | 4,281 |

**关键认知**：
- 训练集 **96% 是合成图**（630K/655K）
- 图片只有 **96×96**，resize 到 448 会严重模糊
- 验证集/测试集是全原图，应**保持独立**

### 1.2 shufazidian 数据集

| 属性 | 数值 |
|------|------|
| 字体类型 | **书法草书（纯草书）** |
| 图片尺寸 | **528×538** |
| 总图数 | **48,628 张** |
| 覆盖字类 | 6,602 个字 |
| 作者数 | 493 位 |

**关键认知**：
- **无合成字**，全部为名家真迹
- 高分辨率（528×538），直接 resize 到 448 保留 85% 细节
- 不区分训练/验证，需按作者 hold out

---

## 二、方案对比：CCC 为主 vs shufazidian 为主

| 维度 | 方案 1：CCC 原图为主 | 方案 2：shufazidian 为主 + CCC 补充 |
|------|---------------------|-----------------------------------|
| **主力数据量** | 42K（25K 训练 + 7K 验证 + 9K 测试） | 43K（443 位训练作者） |
| **主力尺寸** | 96×96 | 528×538 |
| **resize 到 448** | 96→448 **严重模糊** | 528→448 **保留 85% 细节** |
| **平均每字** | 6.5 张 | 6.5 张（但质量高 10 倍） |
| **字体纯度** | 各种字体混合 | **纯草书** |
| **合成比例** | 训练集 96% 合成 | **0% 合成** |
| **模型学什么** | 模糊字形 + 合成痕迹 | 清晰草书笔画 + 真迹风格 |
| **结论** | ❌ **不推荐** | ✅ **推荐** |

---

## 三、推荐方案：shufazidian 为主 + CCC 补充（修正版）

### 3.1 训练数据构成

```
训练数据总量：~265K
├── shufazidian 原图 43K（核心，占输入量 16%）
│   ├── oversampling: 按作者平方根平衡
│   ├── augmentation: 4-5x
│   └── loss weight: 2.0
│
└── CCC 训练集原图 25K（补充，占输入量 9%）
    ├── 筛选: 仅 shufazidian 未覆盖的字类
    ├── augmentation: 1-2x
    └── loss weight: 0.3
```

**最终训练样本**（增强后）：

| 来源 | 增强前 | 增强倍数 | 增强后 | 占比 |
|------|--------|---------|--------|------|
| shufazidian | 43K | 4-5x | **200K-215K** | ~80% |
| CCC 原图 | 25K | 1-2x | **25K-50K** | ~20% |
| **合计** | **68K** | — | **~265K** | **100%** |

### 3.2 目录结构

```
/root/sj-tmp/datasets/CaoshuHQ_v4/
├── source/                          ← 黄金源头，永不动
│   ├── shufazidian/                # 528×538 PNG，48,628 张
│   └── ccc/                        # CCC 原始数据，673K 张
│
├── processed/
│   ├── train/
│   │   ├── shufazidian_hq/         # 443作者 528→448
│   │   └── ccc_original_only/      # 仅训练集原图25K 96→448
│   ├── val/
│   │   ├── shufazidian_block/      # 50作者 528→448（严格隔离）
│   │   ├── ccc_val/                # CCC 验证集 7,724（独立）
│   │   └── ccc_test/               # CCC 测试集 9,548（独立）
│   └── synthetic/                  # 实时增强缓存（可清空重建）
│       └── shufazidian_aug/        # 4-5x 增强样本
│
└── whole_images/                    # 整幅书法图
    ├── source/                     # 1,020 张原始整幅图
    └── end2end_val/                # YOLO切字 + 人工标注
```

### 3.3 shufazidian 处理链路（质量最优）

```python
from PIL import Image

def process_shufazidian(src_path, dst_path):
    img = Image.open(src_path).convert('RGB')  # 528×538
    # 直接到 448，不经过 224
    img = img.resize((448, 448), Image.LANCZOS)
    img.save(dst_path, quality=95)
```

### 3.4 CCC 处理链路（辅助，降权）

```python
def process_ccc(src_path, dst_path):
    img = Image.open(src_path).convert('RGB')  # 96×96
    # 同样到 448，但接受模糊（通过 loss weight 补偿）
    img = img.resize((448, 448), Image.LANCZOS)
    img.save(dst_path, quality=95)
```

### 3.5 增强策略（shufazidian 专用）

```python
from torchvision import transforms

shufa_aug = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomRotation(15),              # 草书旋转容忍度
    transforms.RandomHorizontalFlip(p=0.3),     # 偶尔翻转
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.1
    ),                                          # 墨色变化
    transforms.RandomAffine(
        degrees=0, translate=(0.1, 0.1)
    ),                                          # 轻微位移
    transforms.RandomErasing(p=0.2),            # 模拟印章遮挡
])
```

### 3.6 采样与 Loss Weight 策略

| 策略 | 实现 | 目的 |
|------|------|------|
| **作者分层采样** | `weight = 1/sqrt(author_freq)` | 平衡高产作者与小作者 |
| **字频 Loss Weight** | 低频字 weight↑，高频字 weight↓ | 防止长尾字被淹没 |
| **数据源 Loss Weight** | shufazidian=2.0，CCC=0.3 | 高质量数据主导梯度 |
| **Oversampling** | 低频作者/低频字重复采样 | 确保每字/每作者都被见到 |

**综合 Loss Weight 公式**：
```python
final_weight = base_weight * author_balance * char_freq_weight * source_weight

# 示例：
# shufazidian 低频字：1.0 * 2.0 * 1.5 * 2.0 = 6.0
# CCC 高频字：1.0 * 1.0 * 0.5 * 0.3 = 0.15
```

---

## 四、关键问题讨论

### 4.1 CCC 验证集/测试集能否进训练？

**答案：不能。**

- 验证集 7,724 + 测试集 9,548 = 17,272 张原图
- 如果混入训练：
  - 失去独立评估基准
  - 无法与历史 53.83% 指标对比
  - 早停依据失效

**处理**：验证集/测试集保持独立，仅用于最终评估。

### 4.2 CCC 的 96×96 强制到 448 会污染模型吗？

**风险存在，但可控。**

| 措施 | 效果 |
|------|------|
| loss weight 0.3 | CCC 梯度贡献仅为 shufazidian 的 15% |
| 仅补充未覆盖字类 | CCC 只教"新字"，不干扰"已知字"的草书特征 |
| 不 augment CCC | 避免在模糊基础上再扭曲 |

### 4.3 265K 训练量够吗？

| 对比 | 旧方案 | 新方案 |
|------|--------|--------|
| 总量 | 974K | 265K |
| 合成比例 | 78% | <20% |
| 平均每字 | 93 张 | 40 张 |

**结论**：40 张/字对草书偏少，但**质量远高于数量**。

**应对策略**：
- 训练步数从 100K 降至 **30K-50K**
- 评估间隔降至 **200 步**
- 早停耐心 **3 次**
- 如果 30K 步内不收敛 → 逐步增加 CCC 合成数据（精选 50K-100K）

### 4.4 字类重叠度需要先算吗？

**建议：先算。**

```python
# 快速计算重叠
shufa_chars = set(shufazidian_labels['字'])
ccc_chars = set(ccc_train_labels['字'])

overlap = shufa_chars & ccc_chars           # 两者都有的字
shufa_only = shufa_chars - ccc_chars        # 仅 shufazidian 有
ccc_only = ccc_chars - shufa_chars          # 仅 CCC 有

print(f"重叠字: {len(overlap)}")
print(f"仅 shufazidian: {len(shufa_only)}")
print(f"仅 CCC: {len(ccc_only)}")
```

**决策**：
- 如果重叠 >80%：CCC 补充价值低，可大幅减少 CCC 用量
- 如果重叠 <50%：CCC 补充价值高，需保留更多 CCC 原图

---

## 五、实施路线图

```
Day 1（今天）
├── 1. 建 CaoshuHQ_v4 目录结构
├── 2. 计算字类重叠度（shufazidian vs CCC）
├── 3. shufazidian 443 训练作者 528→448 处理
└── 4. CCC 训练集原图筛选（仅未覆盖字类）+ 96→448

Day 2
├── 5. 生成增强数据 200K（shufazidian 4-5x）
├── 6. 验证集隔离确认（50 作者 + CCC val/test）
└── 7. 修改 train.py（多数据源 + loss weight + 分层采样）

Day 3
├── 8. 启动训练（30K 步目标，早停耐心 3）
└── 9. 监控 shufazidian_block 北极星指标

Day 4-7
├── 10. 训练完成 → 评估真实基线
└── 11. 对比旧模型（CCC/Val 53.83% vs 新模型真实基线）
```

---

## 六、快速决策

| 问题 | 决策 |
|------|------|
| CCC 验证集/测试集进训练？ | ❌ **不进**，保持独立 |
| CCC 合成图（630K）用吗？ | ⚠️ 先不用，30K 步不收敛再加 50-100K 精选 |
| shufazidian 增强倍数？ | **4-5x**（200K-215K） |
| CCC loss weight？ | **0.3**（shufazidian 的 15%） |
| 总训练步数？ | **30K-50K**（早停） |
| 目标验证集？ | **shufazidian_block**（50 作者，无泄漏） |

---

*文档版本：v4.0*  
*日期：2026-05-15*  
*核心变更：确认单字仅两处源头，以 shufazidian 真迹为主，CCC 原图为辅，严格隔离验证集*
