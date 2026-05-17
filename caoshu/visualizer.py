import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Union
import json

class YoloVisualizer:
    """在原图上显示 YOLO 分割框（纯净版本，无文字）"""
    
    def __init__(self, model_path: Union[str, Path], conf_thres: float = 0.25):
        self.model = YOLO(str(model_path))
        self.conf_thres = conf_thres
        print(f"[Visualizer] 加载 YOLO 模型: {model_path}")
    
    def detect_and_visualize(self,
                           image_path: Union[str, Path],
                           output_path: Union[str, Path] = None,
                           box_color: Tuple[int, int, int] = (0, 0, 255),  # 红色
                           thickness: int = 3,
                           imgsz: int = 1344) -> Tuple[np.ndarray, List[Dict]]:
        """
        检测并在原图上画框（无标签、无统计、纯净框）
        Args:
            imgsz: YOLO 推理分辨率长边限制，内部自动 letterbox（防止OOM + 保长宽比）
        """
        image_path = Path(image_path)

        # 读取原图
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 执行检测（imgsz 限制长边，内部 letterbox，bbox 自动映射回原图坐标）
        results = self.model(img, imgsz=imgsz, conf=self.conf_thres, verbose=False)

        # 绘制结果 - 只画框，无任何文字
        vis_img = img.copy()
        boxes_data = []

        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])

            # 只画红色框，不添加任何文字标签
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, thickness)

            boxes_data.append({
                "id": idx + 1,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 3),
                "center": [int((x1+x2)/2), int((y1+y2)/2)]
            })

        # 保存结果（无任何文字 overlay）
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis_img)
            print(f"[Visualizer] 结果已保存: {output_path}")

            # 同时保存 JSON 坐标
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(boxes_data, f, ensure_ascii=False, indent=2)
            print(f"[Visualizer] 坐标已保存: {json_path}")

        return vis_img, boxes_data
    
    def visualize_folder(self, 
                        input_dir: Union[str, Path], 
                        output_dir: Union[str, Path],
                        extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.bmp')):
        """批量可视化文件夹内所有图片"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        print(f"[Visualizer] 找到 {len(image_files)} 张图片")
        
        for img_path in image_files:
            # 新文件名规则：原文件名_result.jpg
            out_path = output_dir / f"{img_path.stem}_result.jpg"
            try:
                self.detect_and_visualize(img_path, out_path)
                print(f"  ✓ 处理完成: {img_path.name} -> {out_path.name}")
            except Exception as e:
                print(f"  ✗ 处理失败: {img_path.name}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO 分割可视化工具（纯净框版本）")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--folder", type=str, help="批量处理文件夹")
    parser.add_argument("--model", type=str, default="params/best.pt", help="YOLO 模型路径")
    parser.add_argument("--output", type=str, default="visualization_output", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    
    args = parser.parse_args()
    
    viz = YoloVisualizer(args.model, conf_thres=args.conf)
    
    if args.image:
        # 单张图片：自动添加 result_ 前缀
        img_path = Path(args.image)
        output_name = f"{img_path.stem}_result.jpg"
        viz.detect_and_visualize(img_path, Path(args.output) / output_name)
    elif args.folder:
        viz.visualize_folder(args.folder, args.output)
    else:
        print("请提供 --image 或 --folder 参数")
