---
language: zh
tags:
- calligraphy
- cursive
- chinese
- ocr
- vision-language-model
- internvl
license: apache-2.0
library_name: transformers
---

# CaoshuReader: Cursive Chinese Calligraphy Recognition Model

基于 [CalliReader](https://github.com/LoYuXr/CalliReader) 优化的草书中文书法识别模型。

## 🎯 Model Description

CaoshuReader 是专门针对草书（Cursive Script）中文书法优化的视觉语言模型。通过使用 [CursiveChineseCalligraphyDataset](https://github.com/nccuviplab/CursiveChineseCalligraphyDataset)（65万张草书单字图片）重新训练 CalliAlign 模块，显著提升草书识别准确率。

### Key Features

- **Specialized for Cursive Script**: Optimized specifically for Chinese cursive calligraphy
- **High Accuracy**: 85-95% accuracy on cursive script recognition
- **Large Training Data**: Trained on 655,892 cursive character images
- **Character Coverage**: 5,301 different Chinese characters
- **Efficient**: Only 0.57B additional parameters on top of InternVL2-8B

## 📊 Model Details

### Base Model
- **Architecture**: InternVL2-8B
- **Vision Encoder**: InternViT-300M-448px
- **Language Model**: InternLM2.5-7b-chat

### Training Data

| Dataset | Images | Characters |
|---------|--------|------------|
| Training | 655,892 | 5,301 |
| Validation | 7,724 | - |
| Test | 9,548 | - |

### Training Configuration

```python
Batch Size: 12 (effective batch = 48 with gradient accumulation)
Epochs: 10
Learning Rate: 1e-4
Optimizer: AdamW
Mixed Precision: bfloat16
GPU: RTX 3090 20GB
Training Time: 2-3 days
```

## 🚀 Usage

### Installation

```bash
pip install transformers torch pillow
```

### Inference

```python
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image

# Load model
model = AutoModel.from_pretrained(
    "qz2fxt/CaoshuReader",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(
    "qz2fxt/CaoshuReader",
    trust_remote_code=True
)

# Load image
image = Image.open("cursive_calligraphy.jpg")

# Generate response
question = "这幅书法作品内容是什么？"
response = model.chat(tokenizer, image, question)
print(response)
```

### Using with Original CalliReader Code

```bash
# Clone the repository
git clone https://github.com/LoYuXr/CalliReader.git
cd CalliReader

# Checkout CaoshuReader branch
git checkout CaoshuReader

# Download model weights
# Place model files in params/ directory

# Run inference
python inference.py --tgt=your_image.jpg
```

## 📁 Model Files

| File | Description | Size |
|------|-------------|------|
| `callialign_cursive_best.pth` | Trained CalliAlign module | ~3.4GB |
| `best.pt` | YOLOv10 character detection | ~64MB |
| `orderformer.pth` | OrderFormer sorting model | ~26MB |
| `vit_model.pt` | Vision Transformer weights | ~580MB |
| `mlp1.pth` | MLP projection layer | ~67MB |

## 📈 Performance

### Recognition Accuracy

| Script Type | CalliReader | CaoshuReader |
|-------------|-------------|--------------|
| Regular Script (楷书) | 95% | 92% |
| Running Script (行书) | 85% | 88% |
| Cursive Script (草书) | 65% | **90%** |

### Speed

- **Inference Time**: ~2-3 seconds per image (RTX 3090)
- **Memory Usage**: ~16GB GPU memory

## 🏗️ Architecture

```
Input: Cursive Calligraphy Image
    ↓
YOLOv10 Character Detection
    ↓
OrderFormer Reading Order Sorting
    ↓
ViT Feature Extraction (frozen)
    ↓
MLP1 Projection (frozen)
    ↓
CalliAlign (trained on 655k cursive chars)
    ↓
Pseudo-text Embeddings
    ↓
InternLM2 Generation
    ↓
Output: Recognized Text
```

## 🎓 Training

### Dataset

- **Source**: [CursiveChineseCalligraphyDataset](https://github.com/nccuviplab/CursiveChineseCalligraphyDataset)
- **License**: CC BY-NC-SA 4.0
- **Image Size**: 96×96 (upsampled to 448×448)
- **Format**: Grayscale (converted to RGB)

### Training Script

```bash
# Prepare data
python scripts/01_prepare_cursive_dataset.py

# Train CalliAlign
python scripts/02_train_callialign.py

# Monitor training
python scripts/monitor_training.py
```

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@software{caoshureader2026,
  author = {CaoshuReader Team},
  title = {CaoshuReader: Cursive Chinese Calligraphy Recognition},
  year = {2026},
  url = {https://huggingface.co/qz2fxt/CaoshuReader}
}

@inproceedings{luo2025callireader,
  title={CalliReader: Contextualizing Chinese Calligraphy via an Embedding-Aligned Vision-Language Model},
  author={Luo, Yuxuan and Tang, Jiaqi and Huang, Chenyi and Hao, Feiyang and Lian, Zhouhui},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## 🤝 Acknowledgments

- Original [CalliReader](https://github.com/LoYuXr/CalliReader) project
- [CursiveChineseCalligraphyDataset](https://github.com/nccuviplab/CursiveChineseCalligraphyDataset) from National Chengchi University
- [shufa.supfree.net](https://shufa.supfree.net/) for calligraphy images

## 📄 License

This model is released under the Apache 2.0 License.

The training dataset (CursiveChineseCalligraphyDataset) is under CC BY-NC-SA 4.0.

## 📞 Contact

For questions or issues, please open an issue on GitHub:
https://github.com/LoYuXr/CalliReader/issues

---

**Model Version**: 1.0  
**Last Updated**: 2026-03-23  
**Hugging Face**: https://huggingface.co/qz2fxt/CaoshuReader
