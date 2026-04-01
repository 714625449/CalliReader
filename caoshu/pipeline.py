"""
CaoshuReader Pipeline - 整图识别
YOLO分割 → 裁剪单字 → CalliReader识别 → Top-3输出
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

# 添加项目路径
PROJECT_ROOT = '/workspace/CalliReader'
sys.path.insert(0, PROJECT_ROOT)

from models.model import (
    load_vision_model,
    load_mlp1,
    load_perceiver_resampler,
    load_normed_tok_embeddings,
)
from config.configu import DOWNSAMPLE_RATIO
from caoshu.dataset import CaoshuDataset, get_transform
from caoshu.visualizer import YoloVisualizer


def pixel_shuffle(x, scale_factor=0.5):
    """像素重排下采样"""
    n, w, h, c = x.size()
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    new_c = int(c / (scale_factor * scale_factor))
    x = x.reshape(n, new_w, 2, new_h, 2, c)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(n, new_w, new_h, new_c)
    return x


@torch.no_grad()
def get_visual_embed(imgs, vit, mlp1):
    """提取视觉特征（包含pixel_shuffle）"""
    vit_out = vit(imgs).last_hidden_state[:, 1:, :]
    h = w = int(vit_out.shape[1] ** 0.5)
    vit_out = vit_out.view(vit_out.shape[0], h, w, -1)
    vit_out = pixel_shuffle(vit_out, scale_factor=DOWNSAMPLE_RATIO)
    vit_out = vit_out.view(vit_out.shape[0], -1, vit_out.shape[-1])
    vit_out = mlp1(vit_out)
    return vit_out


class CalliReaderPipeline:
    """完整的书法识别Pipeline"""

    def __init__(self,
                 yolo_model_path: str,
                 checkpoint_path: str,
                 data_root: str,
                 conf_thres: float = 0.25,
                 num_layers: int = 4):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Pipeline] 使用设备: {self.device}")

        # 1. 加载YOLO分割模型
        print("[Pipeline] 加载YOLO模型...")
        self.visualizer = YoloVisualizer(yolo_model_path, conf_thres)

        # 2. 加载CalliReader模型组件
        print("[Pipeline] 加载CalliReader模型...")
        self.vit = load_vision_model(location='cuda' if torch.cuda.is_available() else 'cpu')
        self.mlp1 = load_mlp1(DOWNSAMPLE_RATIO, location='cuda' if torch.cuda.is_available() else 'cpu')
        self.resampler = load_perceiver_resampler(checkpoint_path, num_layers=num_layers)

        self.vit.eval()
        self.mlp1.eval()
        self.resampler.eval()

        # 3. 加载字符embedding和映射
        print("[Pipeline] 加载字符映射...")
        self.tok_embeddings = load_normed_tok_embeddings()

        # 从数据集加载idx2char映射
        dataset = CaoshuDataset(data_root, 'Validation', transform=None)
        self.idx2char = dataset.idx2char
        self.num_classes = dataset.num_classes
        print(f"[Pipeline] 字符类别数: {self.num_classes}")

        # 4. 预计算所有字符的embedding（分批避免OOM）
        print("[Pipeline] 预计算字符embeddings...")
        self.all_char_embeds = self._compute_all_char_embeddings()

        # 5. 图像预处理
        self.transform = get_transform('Validation')

        print("[Pipeline] 初始化完成！\n")

    def _compute_all_char_embeddings(self, batch_size=1000):
        """分批计算所有字符的embedding"""
        all_embeds = []
        num_batches = (self.num_classes + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, self.num_classes)
            indices = torch.arange(start_idx, end_idx, device=self.device)
            embeds = self.tok_embeddings(indices)
            all_embeds.append(embeds)

        return torch.cat(all_embeds, dim=0)

    def recognize_single_char(self, char_img: Image.Image, topk: int = 3) -> List[Dict]:
        """识别单个字符，返回Top-K候选"""
        # 预处理
        img_tensor = self.transform(char_img).unsqueeze(0).to(self.device).to(torch.bfloat16)

        # 特征提取
        vit_feats = get_visual_embed(img_tensor, self.vit, self.mlp1)

        # Resampler预测
        pred = self.resampler(vit_feats)

        # 处理3D输出 (B, N, D) -> (B, D)
        if pred.dim() == 3:
            pred = pred.mean(dim=1)

        # 归一化并计算相似度
        pred_norm = F.normalize(pred, dim=-1)
        all_char_embeds_norm = F.normalize(self.all_char_embeds, dim=-1)
        similarities = torch.mm(pred_norm, all_char_embeds_norm.t())

        # Top-K
        values, indices = torch.topk(similarities[0], k=topk, dim=-1)
        probs = F.softmax(values, dim=-1)

        # 构建结果
        results = []
        for idx, prob in zip(indices.detach().cpu().float().numpy(), probs.detach().cpu().float().numpy()):
            char = self.idx2char.get(int(idx), '?')
            results.append({
                'char': char,
                'confidence': float(prob),
                'idx': int(idx)
            })

        return results

    def sort_boxes_reading_order(self, boxes_data: List[Dict], row_threshold: int = 50) -> List[Dict]:
        """按阅读顺序排序：从上到下，从右到左（书法习惯）"""
        # 按center_y分组（行）
        boxes_data.sort(key=lambda x: x['center'][1])

        rows = []
        current_row = [boxes_data[0]]

        for box in boxes_data[1:]:
            if abs(box['center'][1] - current_row[0]['center'][1]) < row_threshold:
                current_row.append(box)
            else:
                rows.append(current_row)
                current_row = [box]
        rows.append(current_row)

        # 每行内按center_x从右到左排序
        sorted_boxes = []
        for row in rows:
            row.sort(key=lambda x: -x['center'][0])  # 降序（右到左）
            sorted_boxes.extend(row)

        return sorted_boxes

    def process_image(self,
                     image_path: str,
                     output_dir: str,
                     topk: int = 3,
                     save_crops: bool = True) -> Dict:
        """处理整图：分割 → 识别 → 输出"""

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Pipeline] 处理图片: {image_path.name}")

        # 1. YOLO分割
        print("  [1/4] YOLO分割...")
        result_img_path = output_dir / f"{image_path.stem}_result.jpg"
        vis_img, boxes_data = self.visualizer.detect_and_visualize(
            image_path, result_img_path
        )
        print(f"    检测到 {len(boxes_data)} 个字符")

        if len(boxes_data) == 0:
            print("    警告: 未检测到任何字符")
            return {'image': image_path.name, 'chars': [], 'text': ''}

        # 2. 按阅读顺序排序
        print("  [2/4] 排序...")
        sorted_boxes = self.sort_boxes_reading_order(boxes_data)

        # 3. 裁剪单字并识别
        print("  [3/4] 识别字符...")
        orig_img = cv2.imread(str(image_path))
        chars_dir = output_dir / 'chars'
        if save_crops:
            chars_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, box in enumerate(sorted_boxes):
            x1, y1, x2, y2 = box['bbox']

            # 裁剪，加少量padding
            pad = 4
            h_img, w_img = orig_img.shape[:2]
            x1c = max(0, x1 - pad)
            y1c = max(0, y1 - pad)
            x2c = min(w_img, x2 + pad)
            y2c = min(h_img, y2 + pad)
            crop = orig_img[y1c:y2c, x1c:x2c]

            if crop.size == 0:
                continue

            # 保存裁剪图
            if save_crops:
                crop_path = chars_dir / f"char_{i+1:03d}.jpg"
                cv2.imwrite(str(crop_path), crop)

            # 转PIL并识别
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            top_candidates = self.recognize_single_char(pil_img, topk=topk)

            best_char = top_candidates[0]['char']
            best_conf = top_candidates[0]['confidence']
            top_str = ' | '.join([
                f"{c['char']}({c['confidence']*100:.0f}%)" for c in top_candidates
            ])
            print(f"    char_{i+1:03d}: {top_str}")

            results.append({
                'id': i + 1,
                'bbox': box['bbox'],
                'center': box['center'],
                'yolo_confidence': box['confidence'],
                'top_candidates': top_candidates,
                'best_char': best_char,
                'best_confidence': best_conf,
            })

        # 4. 输出结果
        print("  [4/4] 保存结果...")

        # result.json
        json_path = output_dir / 'result.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'image': image_path.name,
                'total_chars': len(results),
                'chars': results
            }, f, ensure_ascii=False, indent=2)
        print(f"    JSON: {json_path}")

        # result.txt
        text = ''.join([r['best_char'] for r in results])
        txt_path = output_dir / 'result.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        print(f"    TXT:  {txt_path}")
        print(f"    识别结果: {text}")

        return {
            'image': image_path.name,
            'total_chars': len(results),
            'chars': results,
            'text': text
        }


def main():
    parser = argparse.ArgumentParser(description="CalliReader Pipeline - 整图识别")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--ckpt", type=str, required=True, help="CalliReader checkpoint路径")
    parser.add_argument("--yolo", type=str, default="params/best.pt", help="YOLO模型路径")
    parser.add_argument("--data_root", type=str,
                       default="/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset",
                       help="数据集根目录（用于加载idx2char）")
    parser.add_argument("--output", type=str, default="outputs/pipeline", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO置信度阈值")
    parser.add_argument("--topk", type=int, default=3, help="Top-K候选数")
    parser.add_argument("--num_layers", type=int, default=4, help="Resampler层数")
    args = parser.parse_args()

    # 初始化Pipeline
    pipeline = CalliReaderPipeline(
        yolo_model_path=args.yolo,
        checkpoint_path=args.ckpt,
        data_root=args.data_root,
        conf_thres=args.conf,
        num_layers=args.num_layers
    )

    # 处理图片
    result = pipeline.process_image(
        image_path=args.image,
        output_dir=args.output,
        topk=args.topk
    )

    print("\n" + "="*60)
    print("识别完成！")
    print(f"图片: {result['image']}")
    print(f"字符数: {result['total_chars']}")
    print(f"结果: {result['text']}")
    print("="*60)


if __name__ == "__main__":
    main()

