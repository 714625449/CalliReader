import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Union
import json

class YoloVisualizer:
    """在原图上显示 YOLO 分割框（用于人工验证分割质量）"""
    
    def __init__(self, model_path: Union[str, Path], conf_thres: float = 0.25):
        self.model = YOLO(str(model_path))
        self.conf_thres = conf_thres
        print(f"[Visualizer] 加载 YOLO 模型: {model_path}")
    
    def detect_and_visualize(self, 
                           image_path: Union[str, Path], 
                           output_path: Union[str, Path] = None,
                           show_conf: bool = True,
                           show_id: bool = True,
                           box_color: Tuple[int, int, int] = (0, 0, 255),
                           thickness: int = 3) -> Tuple[np.ndarray, List[Dict]]:
        """
        检测并在原图上画框
        """
        # 读取原图
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 执行检测
        results = self.model(img, conf=self.conf_thres, verbose=False)
        
        # 绘制结果
        vis_img = img.copy()
        boxes_data = []
        
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            
            # 画框（红色，3像素宽，适合高分辨率书法图）
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, thickness)
            
            # 构建标签
            labels = []
            if show_id:
                labels.append(f"#{idx+1}")
            if show_conf:
                labels.append(f"{conf:.2f}")
            
            if labels:
                label_text = ":".join(labels)
                # 避免文字超出图像顶部
                text_y = max(y1 - 10, 20)
                cv2.putText(vis_img, label_text, (x1, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            
            boxes_data.append({
                "id": idx + 1,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 3),
                "center": [int((x1+x2)/2), int((y1+y2)/2)]
            })
        
        # 添加统计信息在左上角
        stats_text = f"Total: {len(boxes_data)} chars"
        cv2.putText(vis_img, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 2)
        
        # 保存
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO 分割可视化工具")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--folder", type=str, help="批量处理文件夹")
    parser.add_argument("--model", type=str, default="params/best.pt", help="YOLO 模型路径")
    parser.add_argument("--output", type=str, default="visualization_output", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    
    args = parser.parse_args()
    
    viz = YoloVisualizer(args.model, conf_thres=args.conf)
    
    if args.image:
        viz.detect_and_visualize(args.image, Path(args.output) / "result.jpg")
    elif args.folder:
        viz.visualize_folder(args.folder, args.output)
    else:
        print("请提供 --image 或 --folder 参数")
