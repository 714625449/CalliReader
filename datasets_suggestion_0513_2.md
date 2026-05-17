# CaoshuReader 数据集重建方案 v3.1（图形质量优先）

> 生成时间：2026-05-13  
> 前提：确认 shufazidian 48K 已混入 CCC_split/Training，82.15% 验证结果不可信  
> 核心原则：**源头保留，合成重建，图形质量最优先**

---

## 一、真实源头梳理

### 1.1 单字源头（仅两处）

| 源头 | 规模 | 格式 | 性质 | 质量 |
|------|------|------|------|------|
| **shufazidian** | 48,074 张 | 528×538 PNG | **纯草书真迹，无合成字** | ⭐⭐⭐⭐⭐ |
| **CursiveChineseCalligraphyDataset** | ~955K (655K+300K) | 各异 | 各种字体单字，**含合成字** | ⭐⭐⭐ |

### 1.2 整幅图源头

| 路径 | 数量 | 用途 |
|------|------|------|
| `/caoshu/imgs/` | ~20 张 | 端到端验证 |
| `/caoshu/imgs/samples/samples/images/` | 1,000 张 | 端到端验证 |
| **合计** | **1,020 张** | Pipeline 真实基线 |

---

## 二、数量分析：243K 够不够？

### 2.1 新旧对比

| 指标 | 旧训练 (CaoshuMerged) | 新方案 (243K) | 问题 |
|------|----------------------|---------------|------|
| 总样本 | 974K | 243K | **-75%** |
| 100K 步重复次数 | 每样本看 ~10 次 | 每样本看 **~40 次** | **严重过拟合** |
| 字类覆盖 | 10,423 | ~10,000 | — |
| 平均每字 | 93 张 | **23 张** | **对草书太少** |

### 2.2 结论

**23 张/字撑不起草书识别**。草书类内差异极大（同一字不同作者写法天差地别），需要更多样本才能学到稳定的字 embedding。

**推荐总量：400K-500K**（平均每字 42-48 张，接近旧训练密度）。

---

## 三、重建目录结构

```
/root/sj-tmp/datasets/CaoshuHQ_v3/
├── source/                          ← 黄金源头，只读，永不动
│   ├── shufazidian/                # 528×538 PNG，48K
│   └── cursive_dataset/            # CursiveChineseCalligraphyDataset，~955K
│
├── processed/                       ← 从源头重新预处理
│   ├── train/
│   │   ├── shufazidian_hq/         # 443作者 528→448（~43K）
│   │   ├── shufazidian_aug/        # 增强生成（~150K）
│   │   └── cursive_selected/       # 筛选后 → 448（~250K）
│   └── val/
│       ├── shufazidian_block/      # 50作者 528→448（~5K，严格隔离）
│       └── cursive_val/            # CursiveChineseCalligraphyDataset 独立验证集
│
└── whole_images/                    # 整幅书法图
    ├── source/                     # 1020张原始整幅图
    └── end2end_val/                # YOLO切字 + 人工标注
```

---

## 四、训练集配比（443K 总量）

| 来源 | 数量 | 占比 | 获取方式 | 质量 |
|------|------|------|---------|------|
| **shufazidian_hq** | 43K | 10% | 443作者 528→448 (LANCZOS) | ⭐⭐⭐⭐⭐ |
| **shufazidian_aug** | 150K | 34% | 3.5x 增强（见下方） | ⭐⭐⭐⭐ |
| **cursive_selected** | 250K | 56% | CursiveChineseCalligraphyDataset 筛选 | ⭐⭐⭐ |
| **合计** | **443K** | **100%** | — | — |

### 4.1 shufazidian 处理链路（质量最优）

```
原始 528×538 PNG
    ↓
Image.open().convert('RGB')
    ↓
img.resize((448, 448), Image.LANCZOS)   ← 直接到 448，不经过 224
    ↓
save as .jpg (quality=95)
    ↓
processed/train/shufazidian_hq/{字}/{作者}_{图ID}.jpg
```

### 4.2 shufazidian 增强策略（3.5x）

从 43K 生成 150K，增强倍数 3.5x：

```python
from torchvision import transforms

aug_transform = transforms.Compose([
    transforms.Resize((448, 448)),           # 已经是448，保险
    transforms.RandomRotation(15),           # 草书旋转容忍度
    transforms.RandomHorizontalFlip(p=0.3),  # 偶尔翻转
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # 墨色变化
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 轻微位移
    # Cutout / RandomErasing：模拟印章遮挡
    transforms.ToTensor(),
])
```

### 4.3 CursiveChineseCalligraphyDataset 筛选规则

1. **每字最多 25 张**（防止高频字垄断）
2. **合成字（gen）占比 < 30%**
3. **优先取原始图，剔除明显非草书样本**（如果可识别）
4. 统一 resize 到 448×448

---

## 五、验证集设计

### 5.1 单字级验证

| 验证集 | 来源 | 规模 | 隔离方式 |
|--------|------|------|---------|
| **shufazidian_block** | 50 位验证作者原始 528 | ~5K | 作者 hold out，绝不进训练 |
| cursive_val | CursiveChineseCalligraphyDataset 独立划分 | ~30K | 字类 hold out |

**北极星指标**：shufazidian_block 的 Top-1（无泄漏，真迹纯草书）。

### 5.2 端到端验证

| 来源 | 数量 | 处理方式 |
|------|------|---------|
| 整幅书法图 | 200 张（从 1020 张中抽） | YOLO 切字 → 人工标注 |
| 标注规模 | ~6,000 字 | 逐字核对，建立真值 |

---

## 六、训练策略调整（数据量减半）

| 参数 | 旧训练 (974K) | 新训练 (443K) | 原因 |
|------|--------------|---------------|------|
| 总步数 | 100K | **30K-50K** | 样本少，避免过拟合 |
| Batch size | 16 | 16 | — |
| Grad accum | 4 | 4 | — |
| 学习率 | 5e-5 | **2e-5** | 更保守，防震荡 |
| 评估间隔 | 500 | **200** | 早停需要更密评估 |
| 早停耐心 | — | **3 次不升** | 数据少，收敛更快 |
| 保存策略 | keep 3 | keep 5 | 更密集保存 best |

---

## 七、实施路线图

```
Day 1: 建目录结构 + shufazidian 443→448 处理（~43K，30min）
Day 2: CursiveChineseCalligraphyDataset 筛选 + 448 处理（~250K，2-3h）
Day 3: shufazidian 增强生成 150K（1h）+ 验证集划分
Day 4: 训练脚本改造（多数据源 + 分层采样）
Day 5: 启动训练，监控 shufazidian_block 指标
Day 6-7: 训练 + 早停，拿到新模型真实基线
Week 2: 端到端验证（200张整图标注）+ 与旧模型对比
```

---

## 八、与旧方案对比

| 维度 | 旧方案 (CaoshuMerged) | 新方案 (CaoshuHQ_v3) |
|------|----------------------|---------------------|
| 总量 | 974K | 443K |
| 真迹占比 | ~22% (gen 78%) | **~43%** (gen 控制 <20%) |
| shufazidian 处理 | 528→224→448（损失信息） | **528→448 直接**（保留 85%） |
| 数据泄漏 | 有（shufazidian 混入 CCC_split） | **无（严格作者 hold out）** |
| 验证可信度 | 不可信（82% 泄漏） | **可信（严格隔离）** |
| 端到端验证 | 无 | **200 张整图人工标注** |

---

*文档版本：v3.1*  
*核心变更：确认单字仅两处源头，调整总量至 443K，严格质量优先*
