# 书法识别系统总结

## 系统总览

`/caoshu/` 项目包含 **两个模型**，形成一套完整的书法识别 pipeline：

| 模型 | 用途 | 训练数据 | 检查点位置 |
|------|------|---------|-----------|
| **CalliReader** | 通用书法VLM（整页OCR、理解、问答） | `/root/sj-tmp/callireader_models` | `InternVL/` + `params/`（软链接） |
| **CaoshuReader (草书)** | 草书单字识别（基于CalliReader微调） | `/root/sj-tmp/datasets/CCC_split` | `/root/sj-tmp/checkpoints/CaoshuReader/` |

---

## 一、CalliReader（基础模型）

### 架构
基于 **InternVL** 的视觉语言模型，核心组件：
- `InternVisionModel` (ViT) — 视觉编码器
- `MLP1` — 视觉特征投影层
- `PerceiverResampler` — 将视觉token压缩为3个learned query
- `Token Embeddings` + `Tokenizer` — 文本嵌入（词表大小 92,553）
- `YOLO (best.pt)` — 用于书法字检测/分割

### 关键文件
- **`inference.py`** — 单图/文件夹推理入口
- **`evaluate.py`** — CalliBench 评测（整页识别、区域OCR、选择题等）
- **`models/model.py`** — 所有组件的加载函数
- **`models/perceiver_resampler.py`** — PerceiverAttention + Resampler 实现
- **`config/configu.py`** — 路径与超参数配置

### 参数文件 (`params/`)
```
best.pt              (64MB)  YOLO检测模型
callialign.pth       (3.2GB) PerceiverResampler主权重
gauss_norm.pth       (758MB) 归一化token embedding
gauss_norm_mu_sigma.pth (371KB) 高斯归一化参数
mlp1.pth             (67MB)  MLP1投影层
new1000_token_embedding.pth (758MB)
orderformer.pth      (26MB)  框排序Transformer
token_embedding.pth  (758MB) 原始token embedding
vit_model.pt         (608MB) ViT视觉编码器
```

---

## 二、CaoshuReader（草书单字识别）

这是基于 CalliReader **微调**出来的草书专用模型，只训练 `PerceiverResampler`，冻结其他所有组件。

### 训练数据 (`CCC_split`)
```
/root/sj-tmp/datasets/CCC_split/
├── Training/     (~10,425 个字符类别，文件夹如 一、丁、七、七2、七3...)
├── Validation/   (~9,302 个类别)
└── Test/         (~4,052 个类别)
```
每个类别是一个文件夹（如 `哀1`、`哀12`），里面是该字的多个草书图像。数据集加载时会去掉末尾数字后缀作为标签。

### 已有检查点
```
/root/sj-tmp/checkpoints/CaoshuReader/
├── caoshu_best.pt        (~6.4GB)  最佳模型
├── caoshu_best_val_1.pt  (~6.4GB)
├── caoshu_best_val_2.pt  (~6.4GB)
├── caoshu_best_val_3.pt  (~6.4GB)
├── caoshu_final.pt       (~6.4GB)  最终模型
├── train.log / train_v2.log
```

### 核心文件
| 文件 | 作用 |
|------|------|
| `caoshu/train.py` | 主训练脚本（支持断点续训、磁盘清理、自动保存最佳） |
| `caoshu/train_finetune.py` | 微调脚本（从checkpoint继续，降低lr） |
| `caoshu/pipeline.py` | **整图识别Pipeline**：YOLO分割 → 裁字 → Resampler → Top-K相似度匹配 |
| `caoshu/test_caoshu.py` | 在Validation/Test集上评测单字识别准确率 |
| `caoshu/dataset.py` | `CaoshuDataset` 数据加载器 + `get_transform()` |
| `caoshu/visualizer.py` | `YoloVisualizer` 检测框可视化 |

### 训练方式
1. 冻结 ViT、MLP1、Token Embeddings
2. 只训练 `PerceiverResampler`（4层，dim=4096）
3. Loss：`alignment_loss` = 1 - cosine_similarity(pred, target_embed)
4. 目标：让视觉特征与对应字符的 token embedding 对齐
5. 推理时：预计算所有字符的 embedding，做矩阵相似度匹配，取 Top-K

### 整图识别流程（`pipeline.py`）
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

---

## 三、项目目录结构

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

---

## 四、快速使用命令

### CalliReader 单图识别
```bash
cd /caoshu
python inference.py --tgt=imgs/2.jpg --prompt="这幅书法作品内容是什么？"
```

### CaoshuReader 整图识别
```bash
cd /caoshu
python caoshu/pipeline.py \
    --image=imgs/2.jpg \
    --ckpt=/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best.pt \
    --data_root=/root/sj-tmp/datasets/CCC_split \
    --output=outputs/pipeline
```

### CaoshuReader 验证集测试
```bash
python test_caoshu.py \
    --ckpt=/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best.pt \
    --data_root=/root/sj-tmp/datasets/CCC_split \
    --split=Validation \
    --batch_size=32
```

### 继续训练草书模型
```bash
cd /caoshu/caoshu
python train.py \
    --data_root=/root/sj-tmp/datasets/CCC_split \
    --save_dir=/root/sj-tmp/checkpoints/CaoshuReader \
    --resume=/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best.pt \
    --total_steps=150000
```
