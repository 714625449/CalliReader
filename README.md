<h2 align="center">
  <b>CaoshuReader: Chinese Cursive Script (草書) Recognition via an Embedding-aligned Vision Language Model</b>
</h2>

<div align="center">
    <img src="https://img.shields.io/badge/Status-In%20Development-orange" alt="Status"/>
    <img src="https://img.shields.io/badge/Base-CalliReader-blue" alt="Based on CalliReader"/>
    <img src="https://img.shields.io/badge/Task-Caoshu%20OCR-green" alt="Caoshu OCR"/>
</div>

---

CaoshuReader is a fine-tuned Vision-Language Model (VLM) forked from [**CalliReader**](https://arxiv.org/pdf/2503.06472), specifically enhanced to recognize and interpret **Caoshu (草書)** — the most fluid and abstract style of Chinese calligraphy.

Built on CalliReader's architecture of slicing priors, embedding alignment, and effective fine-tuning, CaoshuReader further specializes on the unique challenges of cursive script: highly stylized strokes, non-linear layouts, and enormous variation across historical masters. It demonstrates strong performance on Caoshu recognition and understanding, while retaining robust OCR ability on general scenes.

> **Solo project** built by one developer using AI agent collaboration (Kimi Code, Claude Code, OpenClaw).  
> A project of this scope would typically require a small team over several months.

---

## 📬 News
- **2025.03** 🚀 CaoshuReader repository initialized — forked from CalliReader.

---

## How to Use

### 0. Install Dependencies

1. Create a conda environment with Python >= 3.9:
```bash
conda create -n caoshureader python=3.9
conda activate caoshureader
```

2. Install essential dependencies:
```bash
pip install requirements.txt
```

3. Install `flash-attn`:
```bash
pip install flash-attn
```

> ⚠️ `flash-attn` requires a Linux system with CUDA installed. If you encounter issues, download the `.whl` file from [here](https://github.com/Dao-AILab/flash-attention/releases):
> ```bash
> pip install flash_attn-xxx.whl
> ```
> For further issues, refer to the [flash-attention repository](https://github.com/Dao-AILab/flash-attention).

---

### 1. Download Weights

*(Coming soon — weights will be released on HuggingFace)*

Download the model weights and place them in the root folder of the cloned repository:
- Fine-tuned VLM weights (`.safetensors`) → `InternVL/` folder
- Pluggable modules → `params/` folder

---

### 2. Inference

Supported formats: `.jpg`, `.png`

**Single image:**
```bash
python inference.py --tgt=<image path>
```
Result is printed directly in the terminal.

**Folder of images:**
```bash
python inference.py --tgt=<folder path> --save_name=<your save name>
```
Results saved to `./results/<your save name>.json`.

---

### 3. Dataset

*(Coming soon)*

CaoshuReader will be evaluated on a Caoshu-focused dataset derived from CalliReader's [CalliBench](https://huggingface.co/datasets/gtang666/CalliBench), extended with additional Caoshu-specific samples.

Original CalliBench covers: Full-page Recognition, Region-wise OCR, Choice Questions (Author, Style, Layout), Bilingual Interpretation, and Intent Analysis — 3,192 image-annotation samples in total.

---

### 4. Training

*(Coming soon)*

Please refer to the **[train](train/)** folder for training scripts and instructions.

---

### 5. Evaluation

*(Coming soon)*

```bash
python evaluate.py --type=<Eval type> --data=<dataset path> --save_name=<test name>
```

Example:
```bash
python evaluate.py --type=full_page --data=./CaoshuBench --save_name=exp
```

---

## 🙏 Acknowledgements

CaoshuReader is built on top of [**CalliReader**](https://github.com/LoYuXr/CalliReader). Full credit to the original authors:

> Luo, Yuxuan and Tang, Jiaqi and Huang, Chenyi and Hao, Feiyang and Lian, Zhouhui.  
> *CalliReader: Contextualizing Chinese Calligraphy via an Embedding-Aligned Vision-Language Model.*  
> ICCV 2025, pp. 23030–23040.

If you use CalliReader's underlying model or dataset, please cite the original work:

```bibtex
@InProceedings{Luo_2025_ICCV,
    author    = {Luo, Yuxuan and Tang, Jiaqi and Huang, Chenyi and Hao, Feiyang and Lian, Zhouhui},
    title     = {CalliReader: Contextualizing Chinese Calligraphy via an Embedding-Aligned Vision-Language Model},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {23030-23040}
}
```
