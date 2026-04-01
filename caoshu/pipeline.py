"""
CalliReader Pipeline - 最终修复版（使用 InternVL 完整模型）
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
from transformers import AutoModel, AutoTokenizer
from ultralytics import YOLO

# 添加项目路径
PROJECT_ROOT = '/workspace/CalliReader'
sys.path.insert(0, PROJECT_ROOT)

from caoshu.dataset import CaoshuDataset
from caoshu.visualizer import YoloVisualizer


class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class CalliReaderPipeline:
    """使用 InternVL 完整模型的 Pipeline"""

    def __init__(self,
                 yolo_model_path: str,
                 checkpoint_path: str,
                 data_root: str,
                 conf_thres: float = 0.25):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_thres = conf_thres
        
        print(f"[Pipeline] 使用设备: {self.device}")

        # 1. 加载 YOLO
        print("[Pipeline] 加载 YOLO 模型...")
        self.yolo = YOLO(yolo_model_path)
        self.yolo_model_path = yolo_model_path

        # 2. 加载 InternVL 完整模型（关键！和 test_caoshu_safe.py 一致）
        internvl_path = 'InternVL'  # 软链接指向的路径
        print(f"[Pipeline] 加载 InternVL 完整模型: {internvl_path}")
        
        self.model = AutoModel.from_pretrained(
            internvl_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True
        ).eval().cuda()

        # 3. 关键：替换 resampler 权重（和 test_caoshu_safe.py 完全一致）
        print(f"[Pipeline] 加载草书权重: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' not in ckpt:
            raise ValueError("Checkpoint 格式错误，缺少 model_state_dict")
        
        state_dict = ckpt['model_state_dict']
        
        # 找到并替换 resampler
        replaced = False
        for name, module in self.model.named_modules():
            if name == 'resampler' and hasattr(module, 'load_state_dict'):
                # 去除 module. 前缀
                clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                module.load_state_dict(clean_state_dict, strict=True)
                print(f"✓ 成功替换 resampler 权重")
                replaced = True
                break
        
        if not replaced:
            raise RuntimeError("未找到 resampler 模块，无法替换权重")

        # 4. 加载其他组件
        self.tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True)
        
        # 加载字符映射
        print("[Pipeline] 加载字符映射...")
        dataset = CaoshuDataset(data_root, 'Validation', transform=None)
        self.idx2char = dataset.idx2char
        self.num_classes = len(self.idx2char)
        print(f"[Pipeline] 字符类别数: {self.num_classes}")

        # 预计算字符 embeddings
        from models.model import load_normed_tok_embeddings
        self.tok_embeddings = load_normed_tok_embeddings(location='cpu')
        self.tok_embeddings = self.tok_embeddings.to(self.device).to(torch.bfloat16)
        self.tok_embeddings.eval()
        
        self.all_char_embeds = self._compute_all_char_embeddings()
        self.all_char_embeds_norm = F.normalize(self.all_char_embeds, dim=-1)

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

    def extract_features(self, img_tensor):
        """使用 InternVL 完整模型提取特征"""
        with torch.no_grad():
            # InternVL 的特征提取方式（参考模型内部实现）
            # 通常是：pixel_values -> vision_model -> resampler
            vit_out = self.model.vision_model(img_tensor).last_hidden_state
            
            # Resampler 处理
            pred = self.model.resampler(vit_out)
            
            # 处理输出维度
            if pred.dim() == 3:
                pred = pred.mean(dim=1)
            
            return pred

    def recognize_single_char(self, char_img: Image.Image, topk: int = 3, verbose: bool = False) -> List[Dict]:
        """识别单个字符"""
        # 确保 RGB
        if char_img.mode != 'RGB':
            char_img = char_img.convert('RGB')
        
        # Resize 到 448（InternVL 通常用这个尺寸）
        if char_img.size != (448, 448):
            char_img = char_img.resize((448, 448), Image.BILINEAR)
        
        # ToTensor 和 Normalize
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(char_img).unsqueeze(0).to(self.device).to(torch.bfloat16)
        
        if verbose:
            print(f"    [调试] 输入: shape={img_tensor.shape}, range=[{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

        with torch.no_grad():
            # 提取特征（使用 InternVL 完整模型）
            pred = self.extract_features(img_tensor)
            
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
        """YOLO 检测"""
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
        """按阅读顺序排序"""
        if not boxes_data:
            return []
        
        boxes_data.sort(key=lambda x: x['center'][1])
        
        rows = []
        current_row = [boxes_data[0]]
        
        for box in boxes_data[1:]:
            if abs(box['center'][1] - current_row[0]['center'][1]) < row_threshold:
                current_row.append(box)
            else:
                current_row.sort(key=lambda x: x['center'][0])
                rows.append(current_row)
                current_row = [box]
        
        current_row.sort(key=lambda x: x['center'][0])
        rows.append(current_row)
        
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

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Pipeline] 处理图片: {image_path.name}")

        # 1. YOLO 分割
        print("  [1/4] YOLO 分割...")
        orig_img, boxes_data = self.yolo_detect(str(image_path))
        print(f"    检测到 {len(boxes_data)} 个字符")
        
        if len(boxes_data) == 0:
            print("    警告: 未检测到任何字符")
            return {'image': image_path.name, 'chars': [], 'text': ''}

        # 保存可视化
        viz = YoloVisualizer(self.yolo_model_path, self.conf_thres)
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
        test_boxes = sorted_boxes[:3] if debug else sorted_boxes
        
        for i, box in enumerate(test_boxes):
            x1, y1, x2, y2 = box['bbox']
            pad = 4
            h_img, w_img = orig_img.shape[:2]
            x1c = max(0, x1 - pad)
            y1c = max(0, y1 - pad)
            x2c = min(w_img, x2 + pad)
            y2c = min(h_img, y2 + pad)
            
            crop_rgb = orig_img_rgb[y1c:y2c, x1c:x2c]
            if crop_rgb.size == 0:
                continue

            if save_crops:
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(chars_dir / f"char_{i+1:03d}.jpg"), crop_bgr)

            pil_img = Image.fromarray(crop_rgb)
            top_candidates = self.recognize_single_char(pil_img, topk=topk, verbose=debug)

            best_char = top_candidates[0]['char']
            top_str = ' | '.join([f"{c['char']}({c['confidence']*100:.0f}%)" for c in top_candidates])
            print(f"    char_{i+1:03d}: {top_str}")

            results.append({
                'id': i + 1,
                'bbox': [int(x) for x in box['bbox']],
                'center': [int(x) for x in box['center']],
                'yolo_confidence': float(box['confidence']),
                'top_candidates': top_candidates,
                'best_char': best_char,
            })

        # 4. 保存结果
        print("  [4/4] 保存结果...")
        
        json_path = output_dir / 'result.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'image': str(image_path.name),
                'total_chars': len(results),
                'chars': results
            }, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        text = ''.join([r['best_char'] for r in results])
        txt_path = output_dir / 'result.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        
        print(f"    识别结果: {text}")

        return {
            'image': image_path.name,
            'total_chars': len(results),
            'chars': results,
            'text': text
        }


def main():
    parser = argparse.ArgumentParser(description="CalliReader Pipeline - 最终版（InternVL完整模型）")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--ckpt", type=str, required=True, help="草书 Resampler 权重路径")
    parser.add_argument("--yolo", type=str, default="params/best.pt", help="YOLO 模型路径")
    parser.add_argument("--data_root", type=str,
                       default="/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset",
                       help="数据集根目录")
    parser.add_argument("--output", type=str, default="outputs/pipeline_final", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度阈值")
    parser.add_argument("--topk", type=int, default=3, help="Top-K 候选数")
    parser.add_argument("--debug", action="store_true", help="调试模式（只处理前3个字）")
    
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
