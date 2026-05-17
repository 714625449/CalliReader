# CaoshuReader 显存优化报告

## 1. 问题诊断

| 场景 | 当前状况 | OOM 根因 |
|------|---------|---------|
| **单字训练 (`train.py`)** | 图像固定 resize 到 `448×448` | ① batch_size / grad_accum 过大；② **ViT 中间激活被保留供反向传播到 Resampler**，这是最大头 |
| **验证集测试 (`test_caoshu.py`)** | 同上 | batch_size 过大；ViT 激活保留 |
| **整图 Pipeline (`pipeline.py`)** | YOLO **直接读取原图**检测 | 超高分辨率扫描件导致 YOLO 前向 OOM |

> 注意：单字训练输入已是固定 448×448，**限制输入分辨率对训练无效**。OOM 是模型内部激活显存导致，不是图像过大。

---

## 2. 核心优化原则

### 2.1 冻结模块必须包 `torch.no_grad()`（最大一刀）

`requires_grad=False` 只能阻止参数更新，**不能阻止 autograd 保留中间激活**。Resampler 的输入来自 ViT→MLP1，反向传播需要回溯到 Resampler 输入，因此 ViT 全路径激活会被保留。

用 `torch.no_grad()` 彻底断开梯度图，这一步省下的显存远大于 autocast 本身（ViT 12+ 层激活 >> Resampler 4 层）。

### 2.2 bf16 不需要 `GradScaler`

`GradScaler` 是为了解决 **fp16 梯度下溢**（指数位只有 5 bit）。bf16 指数位和 fp32 一样宽（8 bit），不存在下溢问题，不需要 scaler。

### 2.3 `autocast` 必须显式指定 `dtype=torch.bfloat16`

`torch.cuda.amp.autocast()` 默认是 **fp16**，不是 bf16。不要手动 `imgs.to(torch.bfloat16)`，让 autocast 自动管理支持 op 的 dtype 转换。

### 2.4 `torch.cuda.empty_cache()` 不降低峰值

它只是把 PyTorch reserved 显存还给 CUDA driver，不降低前向/反向的峰值占用。OOM 发生在峰值瞬间，empty_cache 救不了。可以保留作为例行清理，但不要当核心手段。

---

## 3. 代码修改详情

### 3.1 `caoshu/train.py`

**修改点：**
- 删除 `GradScaler` 全部代码
- `get_visual_embed()` 和 `tok_embeddings()` 用 `torch.no_grad()` 包裹
- `autocast` 显式指定 `dtype=torch.bfloat16`
- 输入 `imgs` 不再手动 `.to(torch.bfloat16)`
- 删除内层冗余 `autocast`（`nn.Embedding` 不在白名单，套了不生效）

```python
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ... 模型加载、数据集加载、optimizer 等不变 ...

    step = start_step
    optimizer.zero_grad()
    loss_item = float('inf')

    while step < args.total_steps:
        for imgs, labels in loader:
            if step >= args.total_steps:
                break

            imgs = imgs.to(device)   # fp32，交给 autocast 管理

            # ========== 冻结模块：no_grad + autocast(bf16) ==========
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    vit_feats = get_visual_embed(imgs, vit, mlp1)
            # vit_feats 无 grad_fn，彻底断开梯度图
            # =========================================================

            # ========== 可训练模块：autocast(bf16) + backward ==========
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = resampler(vit_feats)

                chars = [dataset.idx2char[l.item()] for l in labels]
                token_ids = tokenizer(
                    chars,
                    return_tensors='pt',
                    add_special_tokens=False,
                    padding=True,
                    truncation=True,
                    max_length=4,
                ).input_ids[:, 0].to(device)

                # nn.Embedding 不在 autocast 白名单，内层 autocast 冗余
                with torch.no_grad():
                    tgt_embed = tok_embeddings(token_ids)   # 输出 fp32

                loss = alignment_loss(pred, tgt_embed)
            # =========================================================

            loss_item = loss.item()
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(resampler.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            # ... log / save 逻辑不变 ...
```

---

### 3.2 `test_caoshu.py`

**修改点：**
- 测试阶段全部在 `torch.no_grad()` 下
- `autocast(dtype=torch.bfloat16)` 包裹前向
- **出 autocast 后 `.float()`**，避免 bf16 pred 与 fp32 embedding 做 `torch.mm` 炸 dtype

> ⚠️ **踩坑点**：`autocast` 不会在退出上下文后自动转回 fp32，`pred` 出来仍是 bf16。若 `all_embeds_norm` 是 fp32（预计算时存的是 fp32），`torch.mm(bf16, fp32)` 会 `RuntimeError: expected scalar type BFloat16 but found Float`。

```python
for imgs, labels in tqdm(loader):
    imgs = imgs.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            vit_feats = get_visual_embed(imgs, vit, mlp1)
            pred = resampler(vit_feats)

    pred = pred.float()   # ← 关键修正：autocast 退出后显式转 fp32

    if pred.dim() == 3:
        pred = pred.mean(dim=1)
    elif pred.dim() != 2:
        B = pred.size(0)
        pred = pred.view(B, -1)

    pred_norm = torch.nn.functional.normalize(pred, dim=-1)
    similarities = torch.mm(pred_norm, all_embeds_norm.t())   # fp32 @ fp32
    pred_ids = similarities.argmax(dim=-1)

    # ... 统计准确率逻辑不变 ...
```

---

### 3.3 `caoshu/visualizer.py`

**修改点：**
- YOLO 改用原生 `imgsz=` 参数，内部自动 **letterbox + pad**，保持长宽比
- bbox 输出**自动映射回原图坐标系**
- 比手动 `cv2.resize`（会拉伸变形）更精确

```python
class YoloVisualizer:
    def __init__(self, model_path: Union[str, Path], conf_thres: float = 0.25):
        self.model = YOLO(str(model_path))
        self.conf_thres = conf_thres
        print(f"[Visualizer] 加载 YOLO 模型: {model_path}")

    def detect_and_visualize(self,
                           image_path: Union[str, Path],
                           output_path: Union[str, Path] = None,
                           box_color: Tuple[int, int, int] = (0, 0, 255),
                           thickness: int = 3,
                           imgsz: int = 1344) -> Tuple[np.ndarray, List[Dict]]:
        """
        检测并在原图上画框
        Args:
            imgsz: YOLO 推理分辨率长边限制，内部自动 letterbox（防止OOM + 保长宽比）
        """
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # YOLO 内部处理 letterbox，bbox 输出自动映射回原图坐标系
        results = self.model(img, imgsz=imgsz, conf=self.conf_thres, verbose=False)

        vis_img = img.copy()
        boxes_data = []
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, thickness)
            boxes_data.append({
                "id": idx + 1,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 3),
                "center": [int((x1+x2)/2), int((y1+y2)/2)]
            })

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis_img)
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(boxes_data, f, ensure_ascii=False, indent=2)

        return vis_img, boxes_data
```

---

### 3.4 `caoshu/pipeline.py`

**修改点：**
- 暴露 `--yolo_imgsz` 参数
- 读取原图保留在内存中，用于后续**从原图裁字**
- bbox 是原图坐标系，裁字精度无损（仅一次 resize 到 224）

> 之前方案从缩放图裁字会导致两次 resize（原图→缩放图→224），小字特征损失大。现在 YOLO 吃 letterbox 后的图（省显存），但 bbox 映射回原图坐标，从原图裁字（保精度）。

```python
def process_image(self,
                 image_path: str,
                 output_dir: str,
                 topk: int = 3,
                 save_crops: bool = True,
                 debug: bool = False,
                 yolo_imgsz: int = 1344) -> Dict:
    """
    Args:
        yolo_imgsz: YOLO 推理长边分辨率（默认1344，内部 letterbox 保长宽比）
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Pipeline] 处理图片: {image_path.name}")

    # 读取原图（存在内存中，用于后续从原图裁字）
    orig_img_bgr = cv2.imread(str(image_path))
    if orig_img_bgr is None:
        raise ValueError(f"无法读取图像: {image_path}")
    h, w = orig_img_bgr.shape[:2]
    print(f"  [0/4] 原图分辨率: {w}×{h}")

    # 1. YOLO 分割（imgsz 限制长边，内部 letterbox，bbox 输出原图坐标）
    print("  [1/4] YOLO分割...")
    result_img_path = output_dir / f"{image_path.stem}_result.jpg"
    vis_img, boxes_data = self.visualizer.detect_and_visualize(
        image_path, result_img_path, imgsz=yolo_imgsz
    )
    print(f"    检测到 {len(boxes_data)} 个字符")

    if len(boxes_data) == 0:
        return {'image': image_path.name, 'chars': [], 'text': ''}

    # 2. 按阅读顺序排序（不变）
    sorted_boxes = self.sort_boxes_reading_order(boxes_data)

    # 3. 从原图裁字（bbox 是原图坐标，精度无损）
    print("  [3/4] 识别字符...")
    orig_img_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)
    chars_dir = output_dir / 'chars'
    if save_crops:
        chars_dir.mkdir(parents=True, exist_ok=True)

    results = []
    test_boxes = sorted_boxes[:3] if debug else sorted_boxes

    for i, box in enumerate(test_boxes):
        x1, y1, x2, y2 = box['bbox']
        pad = 4
        h_img, w_img = orig_img_rgb.shape[:2]
        x1c = max(0, x1 - pad)
        y1c = max(0, y1 - pad)
        x2c = min(w_img, x2 + pad)
        y2c = min(h_img, y2 + pad)

        crop_rgb = orig_img_rgb[y1c:y2c, x1c:x2c]
        if crop_rgb.size == 0:
            continue

        if save_crops:
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            crop_path = chars_dir / f"char_{i+1:03d}.jpg"
            cv2.imwrite(str(crop_path), crop_bgr)

        # 转 PIL → resize 到 224 → 识别（仅此一次 resize）
        pil_img = Image.fromarray(crop_rgb)
        top_candidates = self.recognize_single_char(pil_img, topk=topk)
        # ... 后续结果保存逻辑不变 ...
```

**命令行参数增加：**

```python
parser.add_argument("--yolo_imgsz", type=int, default=1344,
                   help="YOLO 推理长边分辨率（默认1344，内部letterbox保长宽比）")
```

---

## 4. 不做的优化（避免无效改动）

| 优化点 | 不做原因 |
|-------|---------|
| `GradScaler` | bf16 指数位和 fp32 同宽，不存在下溢，不需要 scaler |
| `torch.cuda.empty_cache()` 当核心手段 | 不降低峰值占用，OOM 救不了 |
| `gradient_checkpointing` on ViT | ViT 已包 `torch.no_grad()`，不存激活，checkpointing 无用 |
| 手动 `cv2.resize` + 坐标映射 | YOLO 原生 `imgsz=` + letterbox 更干净，且自动映射原图坐标 |
| 限制训练输入分辨率 | 训练输入已固定 448×448，限制无效 |

---

## 5. 显存估算（需实测校准）

> 单样本占用取决于具体 ViT 规格（层数、hidden_dim、patch_size），以下方法实测才准。

**在 `test_caoshu.py` 开头插入实测代码：**

```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# 跑一个 batch
imgs, labels = next(iter(loader))
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        vit_feats = get_visual_embed(imgs.to(device), vit, mlp1)
        pred = resampler(vit_feats)

peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak memory for batch_size={args.batch_size}: {peak:.2f} GB")
```

测出峰值后，按 `peak / batch_size` 得单样本占用，反推当前 GPU 能撑的最大 batch_size。

---

## 6. 快速验证命令

```bash
# 1. 继续训练（batch_size 按实测调整）
cd /caoshu/caoshu
python train.py \
    --data_root=/root/sj-tmp/datasets/CCC_split \
    --save_dir=/root/sj-tmp/checkpoints/CaoshuReader \
    --resume=/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best.pt \
    --batch_size=8 \
    --grad_accum=32

# 2. 验证集测试（batch_size 按实测调整）
python test_caoshu.py \
    --ckpt=/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best.pt \
    --data_root=/root/sj-tmp/datasets/CCC_split \
    --batch_size=16

# 3. 整图识别（YOLO 长边限制 1344）
python caoshu/pipeline.py \
    --image=imgs/2.jpg \
    --ckpt=/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best.pt \
    --data_root=/root/sj-tmp/datasets/CCC_split \
    --output=outputs/pipeline \
    --yolo_imgsz=1344
```

---

## 7. 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `caoshu/train.py` | 去 GradScaler；`no_grad` 包 ViT/MLP1/tok_embeddings；`autocast(dtype=torch.bfloat16)`；去内层冗余 autocast |
| `test_caoshu.py` | `no_grad` + `autocast(dtype=torch.bfloat16)`；出 autocast 后 `.float()` |
| `caoshu/visualizer.py` | `detect_and_visualize()` 支持 `imgsz=` 参数，调用 YOLO 原生 letterbox |
| `caoshu/pipeline.py` | 暴露 `--yolo_imgsz`；从原图裁字（YOLO bbox 已是原图坐标） |
