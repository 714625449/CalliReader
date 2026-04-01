
"""
CalliReader Pipeline - 整图识别（完整修复版 v2）
修复重点：图像预处理与 test_caoshu_safe.py 完全一致
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# 添加项目路径
PROJECT_ROOT = '/workspace/CalliReader'
sys.path.insert(0, PROJECT_ROOT)

from models.model import (
    load_vision_model,
    load_mlp1,
    load_perceiver_resampler,
    load_normed_tok_embeddings,
    load_tokenizer,
)
from config.configu import DOWNSAMPLE_RATIO
from caoshu.dataset import CaoshuDataset, get_transform


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
    """提取视觉特征（与 test_caoshu_safe.py 完全一致）"""
    vit_out = vit(imgs).last_hidden_state[:, 1:, :]
    h = w = int(vit_out.shape[1] ** 0.5)
    vit_out = vit_out.view(vit_out.shape[0], h, w, -1)
    vit_out = pixel_shuffle(vit_out, scale_factor=DOWNSAMPLE_RATIO)
    vit_out = vit_out.view(vit_out.shape[0], -1, vit_out.shape[-1])
    vit_out = mlp1(vit_out)
    return vit_out


class CalliReaderPipeline:
    """完整的书法识别Pipeline（修复版）"""

    def __init__(self,
                 yolo_model_path: str,
                 checkpoint_path: str,
                 data_root: str,
                 conf_thres: float = 0.25,
                 num_layers: int = 4):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Pipeline] 使用设备: {self.device}")

        # 加载YOLO分割模型
        from ultralytics import YOLO
        print("[Pipeline] 加载YOLO模型...")
        self.yolo = YOLO(yolo_model_path)
        self.yolo_model_path = yolo_model_path
        self.conf_thres = conf_thres

        # 加载CalliReader模型组件
        print("[Pipeline] 加载CalliReader模型...")
        self.vit = load_vision_model(location='cuda' if torch.cuda.is_available() else 'cpu')
        self.mlp1 = load_mlp1(DOWNSAMPLE_RATIO, location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载Resampler
        self.resampler = load_perceiver_resampler(None, num_layers=num_layers)
        
        # 加载checkpoint（修复module.前缀）
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                self.resampler.load_state_dict(state_dict)
                print(f"[Pipeline] 加载checkpoint: {checkpoint_path} (step {ckpt.get('total_step', 'unknown')})")
        
        # 关键：全部设为eval模式并冻结
        self.vit.eval()
        self.mlp1.eval()
        self.resampler.eval()
        
        for p in self.vit.parameters():
            p.requires_grad = False
        for p in self.mlp1.parameters():
            p.requires_grad = False
        for p in self.resampler.parameters():
            p.requires_grad = False

        # 加载字符embedding
        print("[Pipeline] 加载字符映射...")
        self.tok_embeddings = load_normed_tok_embeddings(location='cpu')
        self.tok_embeddings = self.tok_embeddings.to(self.device).to(torch.bfloat16)
        self.tok_embeddings.eval()
        
        # 从数据集加载idx2char映射
        dataset = CaoshuDataset(data_root, 'Validation', transform=None)
        self.idx2char = dataset.idx2char
        self.num_classes = len(self.idx2char)
        print(f"[Pipeline] 字符类别数: {self.num_classes}")

        # 预计算所有字符的embedding
        print("[Pipeline] 预计算字符embeddings...")
        self.all_char_embeds = self._compute_all_char_embeddings()
        self.all_char_embeds_norm = F.normalize(self.all_char_embeds, dim=-1)

        # 图像预处理（使用与训练完全一致的transform）
        self.transform = get_transform('Validation')

        # 加载tokenizer
        self.tokenizer = load_tokenizer()

        print("[Pipeline] 初始化完成！\n")

    def _compute_all_char_embeddings(self, batch_size=1000):
        """分批计算所有字符的embedding"""
        self.tok_embeddings.eval()
        all_embeds = []
        num_batches = (self.num_classes + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, self.num_classes)
                indices = torch.arange(start_idx, end_idx, device=self.device)
                embeds = self.tok_embeddings(indices)
                all_embeds.append(embeds)
                if i % 10 == 0:
                    print(f"  预计算进度: {end_idx}/{self.num_classes}")

        return torch.cat(all_embeds, dim=0)

    def recognize_single_char(self, char_img: Image.Image, topk: int = 3, verbose: bool = False) -> List[Dict]:
        """
        识别单个字符（关键修复：与CaoshuDataset处理完全一致）
        """
        # 关键修复1：确保RGB模式
        if char_img.mode != 'RGB':
            char_img = char_img.convert('RGB')
        
        # 关键修复2：确保224x224（CaoshuDataset的输出尺寸）
        if char_img.size != (224, 224):
            char_img = char_img.resize((224, 224), Image.BILINEAR)
        
        # 关键修复3：应用transform（ToTensor + Normalize）
        img_tensor = self.transform(char_img)
        
        # 调试信息（可选）
        if verbose:
            print(f"    [调试] Transform后: shape={img_tensor.shape}, range=[{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        
        # 添加batch维度并转到设备
        img_tensor = img_tensor.unsqueeze(0).to(self.device).to(torch.bfloat16)
        
        # 确保模型在eval模式
        self.vit.eval()
        self.mlp1.eval()
        self.resampler.eval()
        
        with torch.no_grad():
            # 特征提取
            vit_feats = get_visual_embed(img_tensor, self.vit, self.mlp1)
            
            # Resampler预测
            pred = self.resampler(vit_feats)
            
            # 处理3D输出 (B, N, D) -> (B, D)
            if pred.dim() == 3:
                pred = pred.mean(dim=1)
            
            # 归一化并计算相似度
            pred_norm = F.normalize(pred, dim=-1)
            similarities = torch.mm(pred_norm, self.all_char_embeds_norm.t())
            
            if verbose:
                print(f"    [调试] 相似度范围: [{similarities.min():.3f}, {similarities.max():.3f}]")
            
            # Top-K
            values, indices = torch.topk(similarities[0], k=topk, dim=-1)
            probs = F.softmax(values, dim=-1)
            
            # 解析结果
            results = []
            for idx, prob in zip(indices.detach().cpu().float().numpy(), 
                                probs.detach().cpu().float().numpy()):
                char = self.idx2char.get(int(idx), '?')
                results.append({
                    'char': char,
                    'confidence': float(prob),
                    'idx': int(idx)
                })
        
        return results

    def yolo_detect(self, image_path: str):
        """YOLO检测并返回框信息"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        results = self.yolo(img, conf=self.conf_thres, verbose=False)
        
        boxes_data = []
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            boxes_data.append({
                'id': idx,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'center': [(x1+x2)//2, (y1+y2)//2]
            })
        
        return img, boxes_data

    def sort_boxes_reading_order(self, boxes_data: List[Dict], row_threshold: int = 50) -> List[Dict]:
        """按阅读顺序排序：从上到下，从左到右"""
        if not boxes_data:
            return []
        
        # 按center_y排序（行）
        boxes_data.sort(key=lambda x: x['center'][1])
        
        rows = []
        current_row = [boxes_data[0]]
        
        for box in boxes_data[1:]:
            if abs(box['center'][1] - current_row[0]['center'][1]) < row_threshold:
                current_row.append(box)
            else:
                # 当前行按x排序（从左到右）
                current_row.sort(key=lambda x: x['center'][0])
                rows.append(current_row)
                current_row = [box]
        
        # 最后一行
        current_row.sort(key=lambda x: x['center'][0])
        rows.append(current_row)
        
        # 合并
        sorted_boxes = []
        for row in rows:
            sorted_boxes.extend(row)
        
        return sorted_boxes

    def process_image(self,
                     image_path: str,
                     output_dir: str,
                     topk: int = 3,
                     save_crops: bool = True,
                     debug: bool = False) -> Dict:
        """处理整图：分割 → 识别 → 输出"""

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Pipeline] 处理图片: {image_path.name}")

        # 1. YOLO分割
        print("  [1/4] YOLO分割...")
        orig_img, boxes_data = self.yolo_detect(str(image_path))
        print(f"    检测到 {len(boxes_data)} 个字符")
        
        if len(boxes_data) == 0:
            print("    警告: 未检测到任何字符")
            return {'image': image_path.name, 'chars': [], 'text': ''}

        # 保存可视化结果
        from caoshu.visualizer import YoloVisualizer
        viz = YoloVisualizer(self.yolo.model_path, self.conf_thres)
        viz.detect_and_visualize(image_path, output_dir / f"{image_path.stem}_result.jpg")

        # 2. 排序
        print("  [2/4] 排序...")
        sorted_boxes = self.sort_boxes_reading_order(boxes_data)

        # 3. 裁剪并识别
        print("  [3/4] 识别字符...")
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        chars_dir = output_dir / 'chars'
        if save_crops:
            chars_dir.mkdir(parents=True, exist_ok=True)

        results = []
        
        # 调试模式只处理前3个
        test_boxes = sorted_boxes[:3] if debug else sorted_boxes
        
        for i, box in enumerate(test_boxes):
            x1, y1, x2, y2 = box['bbox']

            # 裁剪（加padding）
            pad = 4
            h_img, w_img = orig_img.shape[:2]
            x1c = max(0, x1 - pad)
            y1c = max(0, y1 - pad)
            x2c = min(w_img, x2 + pad)
            y2c = min(h_img, y2 + pad)
            
            crop_rgb = orig_img_rgb[y1c:y2c, x1c:x2c]

            if crop_rgb.size == 0:
                print(f"    char_{i+1:03d}: 裁剪失败，跳过")
                continue

            # 保存裁剪图
            if save_crops:
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                crop_path = chars_dir / f"char_{i+1:03d}.jpg"
                cv2.imwrite(str(crop_path), crop_bgr)

            # 转为PIL并识别
            pil_img = Image.fromarray(crop_rgb)
            top_candidates = self.recognize_single_char(
                pil_img, 
                topk=topk, 
                verbose=debug  # 调试模式打印详细信息
            )

            best_char = top_candidates[0]['char']
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
            })

        # 4. 保存结果
        print("  [4/4] 保存结果...")
        
        json_path = output_dir / 'result.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'image': image_path.name,
                'total_chars': len(results),
                'chars': results
            }, f, ensure_ascii=False, indent=2)
        
        text = ''.join([r['best_char'] for r in results])
        txt_path = output_dir / 'result.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        
        print(f"    JSON: {json_path}")
        print(f"    TXT:  {txt_path}")
        print(f"    识别结果: {text}")

        return {
            'image': image_path.name,
            'total_chars': len(results),
            'chars': results,
            'text': text
        }


def main():
    parser = argparse.ArgumentParser(description="CalliReader Pipeline - 完整修复版")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--ckpt", type=str, required=True, help="CalliReader checkpoint路径")
    parser.add_argument("--yolo", type=str, default="params/best.pt", help="YOLO模型路径")
    parser.add_argument("--data_root", type=str,
                       default="/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset",
                       help="数据集根目录")
    parser.add_argument("--output", type=str, default="outputs/pipeline_fixed", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO置信度阈值")
    parser.add_argument("--topk", type=int, default=3, help="Top-K候选数")
    parser.add_argument("--debug", action="store_true", help="调试模式（只处理前3个字，打印详细信息）")
    
    args = parser.parse_args()

    pipeline = CalliReaderPipeline(
        yolo_model_path=args.yolo,
        checkpoint_path=args.ckpt,
        data_root=args.data_root,
        conf_thres=args.conf,
    )

    result = pipeline.process_image(
        image_path=args.image,
        output_dir=args.output,
        topk=args.topk,
        debug=args.debug
    )

    print("\n" + "="*60)
    print("识别完成！")
    print(f"图片: {result['image']}")
    print(f"字符数: {result['total_chars']}")
    print(f"结果: {result['text']}")
    print("="*60)


if __name__ == "__main__":
    main()
