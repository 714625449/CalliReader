# Hugging Face 上传指南

## 方法一：使用脚本上传（推荐）

### 1. 设置环境

```bash
# 安装依赖
pip install huggingface-hub

# 设置代理（如果需要）
source /root/sj-data/Script/SJ-proxy.sh && proxy_on
```

### 2. 运行上传脚本

```bash
cd /workspace/CalliReader/huggingface_upload

# 使用 Token 上传
python upload_to_hf.py --token hf_YOUR_TOKEN_HERE
```

## 方法二：手动上传

### 1. 登录 Hugging Face

```bash
# 安装 CLI
pip install huggingface-hub

# 登录
huggingface-cli login
# 输入你的 Token: hf_YOUR_TOKEN_HERE
```

### 2. 创建仓库

```bash
# 创建模型仓库
huggingface-cli repo create CaoshuReader --type model

# 或者使用 git
huggingface-cli repo create CaoshuReader --type model --organization qz2fxt
```

### 3. 克隆仓库并上传文件

```bash
# 克隆仓库
git clone https://huggingface.co/qz2fxt/CaoshuReader
cd CaoshuReader

# 复制文件
cp /workspace/CalliReader/huggingface_upload/README.md .
cp /workspace/CalliReader/params/callialign_cursive_best.pth .
cp /workspace/CalliReader/params/best.pt .
cp /workspace/CalliReader/params/orderformer.pth .
cp /workspace/CalliReader/params/vit_model.pt .
cp /workspace/CalliReader/params/mlp1.pth .

# 提交并推送
git add .
git commit -m "Initial upload of CaoshuReader model"
git push
```

## 方法三：使用 Python API

```python
from huggingface_hub import login, upload_file, create_repo

# 登录
login(token="hf_YOUR_TOKEN_HERE")

# 创建仓库
create_repo("qz2fxt/CaoshuReader", repo_type="model", exist_ok=True)

# 上传文件
upload_file(
    path_or_fileobj="/workspace/CalliReader/params/callialign_cursive_best.pth",
    path_in_repo="callialign_cursive_best.pth",
    repo_id="qz2fxt/CaoshuReader",
    repo_type="model"
)
```

## 文件清单

需要上传的文件：

| 文件 | 路径 | 大小 |
|------|------|------|
| README.md | ./README.md | ~5KB |
| CalliAlign模型 | ./callialign_cursive_best.pth | ~3.4GB |
| YOLO检测模型 | ./best.pt | ~64MB |
| OrderFormer模型 | ./orderformer.pth | ~26MB |
| ViT模型 | ./vit_model.pt | ~580MB |
| MLP1模型 | ./mlp1.pth | ~67MB |

## 注意事项

1. **Token 权限**: 确保 Token 有 `write` 权限
2. **文件大小**: 大文件需要使用 Git LFS
3. **网络问题**: 如遇网络问题，请检查代理设置
4. **存储空间**: 确保 Hugging Face 账户有足够的存储空间

## 验证上传

上传完成后，访问：
https://huggingface.co/qz2fxt/CaoshuReader

## 故障排除

### SSL 错误
```bash
# 更新证书
pip install --upgrade certifi

# 或设置环境变量
export CURL_CA_BUNDLE=/path/to/certificate.pem
```

### 权限错误
```bash
# 检查 Token 权限
huggingface-cli whoami

# 重新登录
huggingface-cli logout
huggingface-cli login
```

### 大文件上传失败
```bash
# 安装 Git LFS
git lfs install

# 跟踪大文件
git lfs track "*.pth"
git lfs track "*.pt"
```
