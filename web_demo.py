"""
CalliReader / CaoshuReader Gradio Web Demo
Usage:
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    source /root/miniconda3/bin/activate callireader
    python web_demo.py
"""

import os
import sys
import gc
import torch
import gradio as gr

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_caoshu import load_model_with_caoshu_ckpt, single_rec
from inference import single_rec as single_rec_original
from transformers import AutoModel, AutoTokenizer
from ultralytics import YOLO
from config.configu import INTERNVL_PATH, YOLO_CHECKPOINT

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------- Global State ----------
_current_model = None
_current_mode = None

def get_model(mode: str):
    """Lazy-load model with caching."""
    global _current_model, _current_mode

    if _current_model is not None and _current_mode == mode:
        return _current_model

    # Unload previous model to free VRAM
    if _current_model is not None:
        del _current_model
        _current_model = None
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[Demo] Loading model: {mode}")
    if mode == "caoshu":
        model, tokenizer, detect_model, gen_cfg = load_model_with_caoshu_ckpt(
            caoshu_ckpt_path="/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best_val_1.pt",
            num_layers=8,
            num_learns=12,
        )
        # Reduce max_new_tokens to avoid OOM and long hallucinations
        gen_cfg["max_new_tokens"] = 256
        gen_cfg["repetition_penalty"] = 1.3
    else:
        model = AutoModel.from_pretrained(
            INTERNVL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True)
        detect_model = YOLO(str(YOLO_CHECKPOINT))
        gen_cfg = dict(
            num_beams=1,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.3,
        )

    _current_mode = mode
    _current_model = (model, tokenizer, detect_model, gen_cfg)
    print(f"[Demo] Model loaded: {mode}")
    return _current_model


def recognize(image_path, mode, prompt):
    """Run OCR inference."""
    if image_path is None:
        return "请先上传图片。"

    model, tokenizer, detect_model, gen_cfg = get_model(mode)

    try:
        if mode == "caoshu":
            pred = single_rec(
                model, tokenizer, detect_model, gen_cfg, image_path, prompt, use_p=True
            )
        else:
            pred = single_rec_original(
                model, tokenizer, detect_model, gen_cfg, image_path, prompt, use_p=True
            )
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return f"❌ CUDA 显存不足 (OOM)。建议：\n1. 换用 Original 模型\n2. 降低图片分辨率\n3. 缩小检测区域\n\n详情: {str(e)[:200]}"
    except Exception as e:
        return f"❌ 推理出错: {str(e)}"

    return pred


# ---------- Gradio UI ----------
with gr.Blocks(title="CalliReader Web Demo") as demo:
    gr.Markdown("# 🖌️ CalliReader / CaoshuReader 书法 OCR Demo")
    gr.Markdown(
        "上传一张整页书法作品图片，选择模型并输入 Prompt，即可获得 OCR 识别结果。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="filepath", label="上传书法图片", height=400
            )
            mode_radio = gr.Radio(
                choices=[
                    ("Original CalliReader", "original"),
                    ("CaoshuReader (草书 Resampler)", "caoshu"),
                ],
                value="original",
                label="选择模型",
            )
            prompt_input = gr.Textbox(
                value="读出图中所有文字",
                label="Prompt",
                placeholder="例如：读出图中所有文字",
            )
            run_btn = gr.Button("🚀 开始识别", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="识别结果",
                lines=12,
                max_lines=20,
                interactive=False,
            )

    gr.Markdown("""
    ### 使用提示
    - **Original**：4层/3query Resampler，显存占用较小，适合大多数图像。
    - **CaoshuReader**：8层/12query 草书 Resampler，对草书识别有提升，但复杂图像可能 OOM。
    - 如果 CaoshuReader 报 OOM，请换用 Original 模型或降低图片分辨率。
    - 模型切换时会自动释放旧模型显存，首次加载需等待 30~60 秒。
    """)

    run_btn.click(
        fn=recognize,
        inputs=[image_input, mode_radio, prompt_input],
        outputs=output_text,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
