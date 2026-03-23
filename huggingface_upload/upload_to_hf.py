#!/usr/bin/env python3
"""
上传 CaoshuReader 模型到 Hugging Face
使用方法: python upload_to_hf.py --token YOUR_HF_TOKEN
"""

import os
import sys
import argparse
from pathlib import Path

def upload_model(token=None):
    """上传模型到 Hugging Face"""
    
    try:
        from huggingface_hub import login, HfApi, upload_file, upload_folder, create_repo
    except ImportError:
        print("请先安装 huggingface_hub: pip install huggingface-hub")
        return
    
    # 登录
    if token:
        login(token=token)
        print("✅ 已使用提供的 Token 登录")
    else:
        # 尝试从环境变量或缓存登录
        try:
            api = HfApi()
            user_info = api.whoami()
            print(f"✅ 已登录: {user_info['name']}")
        except:
            print("❌ 未登录，请提供 --token 参数")
            return
    
    repo_id = "qz2fxt/CaoshuReader"
    
    # 创建仓库
    print(f"\n创建仓库: {repo_id}")
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✅ 仓库已创建/已存在")
    except Exception as e:
        print(f"⚠️ 仓库创建: {e}")
        return
    
    # 上传 README
    print("\n上传 README.md...")
    try:
        upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )
        print("✅ README.md 上传成功")
    except Exception as e:
        print(f"❌ README.md 上传失败: {e}")
    
    # 上传模型文件
    model_files = [
        ("/workspace/CalliReader/params/callialign_cursive_best.pth", "callialign_cursive_best.pth"),
        ("/workspace/CalliReader/params/best.pt", "best.pt"),
        ("/workspace/CalliReader/params/orderformer.pth", "orderformer.pth"),
        ("/workspace/CalliReader/params/vit_model.pt", "vit_model.pt"),
        ("/workspace/CalliReader/params/mlp1.pth", "mlp1.pth"),
    ]
    
    print("\n上传模型文件...")
    for local_path, repo_path in model_files:
        if os.path.exists(local_path):
            print(f"  上传 {repo_path}...")
            try:
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"  ✅ {repo_path} 上传成功")
            except Exception as e:
                print(f"  ❌ {repo_path} 上传失败: {e}")
        else:
            print(f"  ⚠️ {local_path} 不存在，跳过")
    
    # 上传代码
    print("\n上传代码文件...")
    code_files = [
        "/workspace/CalliReader/inference.py",
        "/workspace/CalliReader/config/configu.py",
        "/workspace/CalliReader/models/model.py",
        "/workspace/CalliReader/models/perceiver_resampler.py",
        "/workspace/CalliReader/utils/utils.py",
    ]
    
    for file_path in code_files:
        if os.path.exists(file_path):
            repo_path = os.path.basename(file_path)
            try:
                upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"code/{repo_path}",
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"  ✅ {repo_path} 上传成功")
            except Exception as e:
                print(f"  ❌ {repo_path} 上传失败: {e}")
    
    print(f"\n{'='*60}")
    print(f"上传完成!")
    print(f"模型地址: https://huggingface.co/{repo_id}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="上传 CaoshuReader 到 Hugging Face")
    parser.add_argument("--token", type=str, help="Hugging Face Token")
    args = parser.parse_args()
    
    upload_model(token=args.token)

if __name__ == "__main__":
    main()
