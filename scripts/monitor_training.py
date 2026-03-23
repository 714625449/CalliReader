#!/usr/bin/env python3
"""
训练监控脚本 - 实时查看训练进度和GPU状态
"""

import os
import time
import subprocess
import re
from datetime import datetime

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        gpus = []
        for line in lines:
            parts = line.split(', ')
            gpus.append({
                'util': float(parts[0]),
                'mem_used': float(parts[1]),
                'mem_total': float(parts[2]),
                'temp': float(parts[3])
            })
        return gpus
    except:
        return []

def parse_log_file(log_path):
    """解析训练日志"""
    if not os.path.exists(log_path):
        return None
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        # 查找最新的训练信息
        latest_loss = None
        latest_lr = None
        current_epoch = None
        
        for line in reversed(lines):
            # 匹配损失
            loss_match = re.search(r'loss[=:]\s*([\d.]+)', line, re.IGNORECASE)
            if loss_match and not latest_loss:
                latest_loss = float(loss_match.group(1))
            
            # 匹配学习率
            lr_match = re.search(r'lr[=:]\s*([\d.e+-]+)', line, re.IGNORECASE)
            if lr_match and not latest_lr:
                latest_lr = float(lr_match.group(1))
            
            # 匹配epoch
            epoch_match = re.search(r'Epoch\s*(\d+)', line, re.IGNORECASE)
            if epoch_match and not current_epoch:
                current_epoch = int(epoch_match.group(1))
            
            if latest_loss and latest_lr and current_epoch:
                break
        
        return {
            'epoch': current_epoch,
            'loss': latest_loss,
            'lr': latest_lr,
            'lines': len(lines)
        }
    except:
        return None

def monitor():
    """监控训练状态"""
    log_path = "/workspace/CalliReader/logs/02_train_callialign.log"
    
    print("=" * 70)
    print("CalliReader 训练监控")
    print("=" * 70)
    print(f"监控日志: {log_path}")
    print("按 Ctrl+C 退出")
    print("=" * 70)
    
    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("=" * 70)
            print(f"CalliReader 训练监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            
            # GPU状态
            gpus = get_gpu_info()
            if gpus:
                print("\n📊 GPU 状态:")
                for i, gpu in enumerate(gpus):
                    mem_pct = gpu['mem_used'] / gpu['mem_total'] * 100
                    print(f"  GPU {i}: {gpu['util']:5.1f}% 利用率 | "
                          f"{gpu['mem_used']:.0f}/{gpu['mem_total']:.0f} MB ({mem_pct:.1f}%) | "
                          f"{gpu['temp']:.0f}°C")
            
            # 训练状态
            log_info = parse_log_file(log_path)
            if log_info:
                print("\n📈 训练状态:")
                print(f"  当前 Epoch: {log_info['epoch']}")
                print(f"  当前 Loss:  {log_info['loss']:.4f}" if log_info['loss'] else "  当前 Loss:  N/A")
                print(f"  学习率:     {log_info['lr']:.2e}" if log_info['lr'] else "  学习率:     N/A")
                print(f"  日志行数:   {log_info['lines']}")
            else:
                print("\n⏳ 等待训练开始...")
            
            # 检查模型文件
            model_path = "/workspace/CalliReader/params/callialign_cursive_best.pth"
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
                print(f"\n💾 最佳模型: {model_path}")
                print(f"  大小: {size_mb:.1f} MB")
                print(f"  更新时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            print("\n" + "=" * 70)
            print("刷新间隔: 5秒")
            print("=" * 70)
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == "__main__":
    monitor()
