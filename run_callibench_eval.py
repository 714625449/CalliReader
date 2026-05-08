#!/usr/bin/env python3
"""
CalliBench 评估脚本 - 简化版
使用 examples 目录下的图片进行评估测试
"""

import os
import sys
import json
import time
import torch
from PIL import Image
from tqdm import tqdm
import argparse

sys.path.append('/workspace/CalliReader')

from transformers import AutoModel, AutoTokenizer
from ultralytics import YOLO
from config.configu import *

class CalliBenchEvaluator:
    """CalliBench 评估器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载模型
        print("\n[1/3] 加载 InternVL 模型...")
        self.model = AutoModel.from_pretrained(
            INTERNVL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda()
        
        print("[2/3] 加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            INTERNVL_PATH, 
            trust_remote_code=True
        )
        
        print("[3/3] 加载 YOLO 检测模型...")
        self.detect_model = YOLO(YOLO_CHECKPOINT)
        
        self.generation_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
        )
        
        print("✅ 模型加载完成!\n")
    
    def evaluate_single_image(self, image_path, prompt="这幅书法作品内容是什么？"):
        """评估单张图片"""
        try:
            start_time = time.time()
            
            response, history = self.model.chat_ocr(
                self.tokenizer,
                self.detect_model,
                image_path,
                prompt,
                self.generation_config,
                use_p=True,
                hard_vq=False,
                drop_zero=False,
                repetition_penalty=1.0,
                return_history=True,
                verbose=False
            )
            
            inference_time = time.time() - start_time
            
            return {
                'success': True,
                'response': response,
                'inference_time': inference_time,
                'image_path': image_path
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path
            }
    
    def run_full_benchmark(self, test_images_dir="./examples"):
        """运行完整基准测试"""
        print("=" * 70)
        print("CalliBench 评估报告")
        print("=" * 70)
        print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模型: InternVL2-8B + CalliAlign")
        print(f"测试图片目录: {test_images_dir}")
        print("=" * 70)
        
        # 获取测试图片
        test_images = [f for f in os.listdir(test_images_dir) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if not test_images:
            print(f"❌ 未找到测试图片: {test_images_dir}")
            return
        
        print(f"\n找到 {len(test_images)} 张测试图片\n")
        
        # 运行测试
        results = []
        total_time = 0
        
        for img_name in tqdm(test_images, desc="评估进度"):
            img_path = os.path.join(test_images_dir, img_name)
            result = self.evaluate_single_image(img_path)
            results.append(result)
            
            if result['success']:
                total_time += result['inference_time']
        
        # 生成报告
        self.generate_report(results, total_time)
        
        return results
    
    def generate_report(self, results, total_time):
        """生成评估报告"""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print("\n" + "=" * 70)
        print("评估结果汇总")
        print("=" * 70)
        
        print(f"\n📊 总体统计:")
        print(f"  总测试数: {len(results)}")
        print(f"  成功: {len(successful)}")
        print(f"  失败: {len(failed)}")
        print(f"  成功率: {len(successful)/len(results)*100:.1f}%")
        
        if successful:
            avg_time = total_time / len(successful)
            print(f"\n⏱️  推理速度:")
            print(f"  总推理时间: {total_time:.2f}秒")
            print(f"  平均推理时间: {avg_time:.2f}秒/张")
            print(f"  吞吐量: {1/avg_time:.2f} 张/秒")
        
        print(f"\n📝 详细结果:")
        print("-" * 70)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}/{len(results)}] {os.path.basename(result['image_path'])}")
            
            if result['success']:
                print(f"  状态: ✅ 成功")
                print(f"  推理时间: {result['inference_time']:.2f}秒")
                print(f"  识别结果: {result['response'][:100]}..." 
                      if len(result['response']) > 100 
                      else f"  识别结果: {result['response']}")
            else:
                print(f"  状态: ❌ 失败")
                print(f"  错误: {result['error']}")
        
        # 保存报告
        report_path = f"outputs/callibench_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("outputs", exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'summary': {
                    'total': len(results),
                    'success': len(successful),
                    'failed': len(failed),
                    'success_rate': len(successful)/len(results)*100,
                    'avg_inference_time': total_time/len(successful) if successful else 0
                },
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 报告已保存: {report_path}")
        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="CalliBench 评估")
    parser.add_argument('--data', type=str, default='./examples',
                       help='测试图片目录')
    parser.add_argument('--output', type=str, default='./outputs',
                       help='输出目录')
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = CalliBenchEvaluator()
    
    # 运行评估
    evaluator.run_full_benchmark(args.data)

if __name__ == "__main__":
    main()
