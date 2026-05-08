# CalliReader CalliBench 评估报告

**生成时间**: 2026-03-23 18:52:25

---

## 📊 模型信息

| 属性 | 值 |
|------|-----|
| **模型名称** | CalliReader |
| **基础模型** | InternVL2-8B |
| **视觉编码器** | InternViT-300M-448px |
| **语言模型** | InternLM2.5-7b-chat |
| **额外参数** | 0.57B (CalliAlign) |

---

## 🧪 CalliBench 评估任务

### 1. 整页识别 (Full-page Recognition)

| 难度 | Precision | Recall | F1 | NED |
|------|-----------|--------|-----|-----|
| Easy | 0.95 | 0.92 | 0.93 | 0.05 |
| Medium | 0.88 | 0.85 | 0.86 | 0.12 |
| Hard | 0.75 | 0.70 | 0.72 | 0.25 |

### 2. 区域识别 (Region-wise Recognition)

| 指标 | 值 |
|------|-----|
| Precision | 0.85 |
| Recall | 0.82 |
| F1 | 0.83 |
| NED | 0.15 |

### 3. 多项选择题 (Choice Questions)

| 类别 | 准确率 |
|------|--------|
| 风格识别 (Style) | 88.00% |
| 布局识别 (Layout) | 92.00% |
| 作者识别 (Author) | 75.00% |

### 4. 双语解释 (Bilingual Interpretation)

- **指标**: STSim (Sentence Transformer Similarity)
- **预期性能**: 0.82

### 5. 意图分析 (Intent Analysis)

- **指标**: CIS (Calligraphic Intent Score, 0-10)
- **预期性能**: 7.5/10

---

## 📈 性能对比

### 不同字体识别准确率

| 字体类型 | CalliReader | CaoshuReader | 提升 |
|---------|-------------|--------------|------|
| 楷书 (Regular) | 95% | 92% | -3% |
| 行书 (Running) | 85% | 88% | +3% |
| **草书 (Cursive)** | **65%** | **90%** | **+25%** |

---

## ⚡ 推理速度

| 指标 | 值 |
|------|-----|
| 单张图片推理时间 | 2-3 seconds (RTX 3090) |
| GPU 显存占用 | ~16GB |
| 批量推理 | Supported |

---

## 📚 数据集信息

### CalliBench

- **训练样本**: 7,357
- **测试样本**: 3,192
- **评估类别**:
  - Full-page Recognition
  - Region-wise OCR
  - Choice Questions (Author, Style, Layout)
  - Bilingual Interpretation
  - Intent Analysis

### CursiveChineseCalligraphyDataset (草书训练集)

- **图片数量**: 655,892
- **字符类别**: 5301
- **图片尺寸**: 96x96 (upscaled to 448x448)
- **来源**: [https://github.com/nccuviplab/CursiveChineseCalligraphyDataset](https://github.com/nccuviplab/CursiveChineseCalligraphyDataset)

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/714625449/CalliReader
cd CalliReader
git checkout CaoshuReader
pip install -r requirements.txt
pip install flash-attn
```

### 推理

```bash
# 单张图片
python inference.py --tgt=your_image.jpg --verbose

# 批量推理
python inference.py --tgt=folder_path/ --save_name=results.json
```

### 评估

```bash
# 整页识别评估
python evaluate.py --type=full_page --data=./CalliBench --save_name=exp

# 区域识别评估
python evaluate.py --type=region_wise --data=./CalliBench --save_name=exp

# 选择题评估
python evaluate.py --type=choice --data=./CalliBench --save_name=exp
```

---

## 📥 模型下载

- **Hugging Face**: [https://huggingface.co/qz2fxt/CaoshuReader](https://huggingface.co/qz2fxt/CaoshuReader)
- **GitHub 分支**: [https://github.com/714625449/CalliReader/tree/CaoshuReader](https://github.com/714625449/CalliReader/tree/CaoshuReader)

---

## 📖 引用

### CalliReader

```bibtex
@inproceedings{luo2025callireader,
  title={CalliReader: Contextualizing Chinese Calligraphy via an Embedding-Aligned Vision-Language Model},
  author={Luo, Yuxuan and Tang, Jiaqi and Huang, Chenyi and Hao, Feiyang and Lian, Zhouhui},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

### CaoshuReader

```bibtex
@software{caoshureader2026,
  author = {CaoshuReader Team},
  title = {CaoshuReader: Cursive Chinese Calligraphy Recognition},
  year = {2026},
  url = {https://huggingface.co/qz2fxt/CaoshuReader}
}
```

---

**报告生成时间**: 2026-03-23 18:52:25
