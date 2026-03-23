# CalliReader 草书训练完整指南

## 📋 执行计划总览

| 阶段 | 任务 | 预计时间 | 状态 |
|------|------|----------|------|
| 0 | 下载 CursiveChineseCalligraphyDataset | 1-2小时 | 🔄 进行中 |
| 1 | 数据预处理 | 2-4小时 | ⏳ 等待 |
| 2 | 训练 CalliAlign | 2-3天 | ⏳ 等待 |
| 3 | 部署测试 | 30分钟 | ⏳ 等待 |

---

## 🚀 快速开始

### 步骤 0: 监控数据集下载

```bash
# 查看下载进度
du -sh /root/sj-tmp/CCCdatabase
ls /root/sj-tmp/CCCdatabase/

# 查看下载日志
tail -f /tmp/clone_ccc.log
```

### 步骤 1: 数据预处理 (下载完成后)

```bash
cd /workspace/CalliReader

# 运行预处理脚本
python scripts/01_prepare_cursive_dataset.py

# 预期输出:
# - /workspace/CalliReader/data/callireader_cursive/single_char/{train,val,test}/
# - /workspace/CalliReader/data/callireader_cursive/page_level/
# - /workspace/CalliReader/data/callireader_cursive/char_mapping.json
```

### 步骤 2: 训练 CalliAlign

```bash
cd /workspace/CalliReader

# 方式1: 直接运行训练脚本
python scripts/02_train_callialign.py

# 方式2: 使用完整流程脚本
chmod +x scripts/run_training_pipeline.sh
./scripts/run_training_pipeline.sh

# 监控训练 (新开终端)
python scripts/monitor_training.py
```

### 步骤 3: 部署测试

```bash
# 训练完成后，测试新模型
cd /workspace/CalliReader
python inference.py --tgt=examples/0.jpg --verbose
```

---

## 📁 文件说明

### 创建的脚本文件

| 文件 | 用途 |
|------|------|
| `scripts/01_prepare_cursive_dataset.py` | 数据预处理和格式转换 |
| `scripts/02_train_callialign.py` | CalliAlign 训练脚本 |
| `scripts/run_training_pipeline.sh` | 完整训练流程 |
| `scripts/monitor_training.py` | 训练监控工具 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `params/callialign_cursive_best.pth` | 最佳模型 |
| `params/callialign_cursive_epoch{N}.pth` | 每轮检查点 |
| `logs/02_train_callialign.log` | 训练日志 |

---

## ⚙️ 训练配置 (RTX 3090 20GB)

```python
batch_size = 12          # 单卡可承受
accumulation_steps = 4   # 有效 batch = 48
num_epochs = 10
lr = 1e-4
mixed_precision = True   # 启用混合精度
```

**显存占用**: 约 16-18GB / 20GB

---

## 🔍 监控训练

### 方式1: 实时监控
```bash
python scripts/monitor_training.py
```

### 方式2: 查看日志
```bash
tail -f /workspace/CalliReader/logs/02_train_callialign.log
```

### 方式3: GPU监控
```bash
watch -n 5 nvidia-smi
```

---

## 🛠️ 故障排除

### 问题1: 显存不足
```bash
# 修改 scripts/02_train_callialign.py
batch_size = 8           # 减小batch size
accumulation_steps = 6   # 增加累积步数
```

### 问题2: 训练中断
```bash
# 脚本会自动保存检查点，重新运行即可恢复
# 如需从特定epoch恢复，修改脚本中的 resume 逻辑
```

### 问题3: 数据集下载慢
```bash
# 可以手动下载后上传到服务器
# 或使用代理加速
export https_proxy=http://your-proxy:port
git clone https://github.com/nccuviplab/CursiveChineseCalligraphyDataset.git
```

---

## 📊 预期结果

### 训练指标
- **训练损失**: 从 ~0.5 降至 ~0.05
- **验证损失**: 稳定在 ~0.08 以下
- **训练时间**: 2-3天 (单卡3090)

### 识别效果对比

| 模型 | 草书识别准确率 |
|------|---------------|
| 原CalliReader | 60-70% |
| 草书训练后 | 85-95% |

---

## 📝 后续步骤

训练完成后，如需进一步提升效果:

1. **e-IT微调**: 使用 CalliTrain 数据进行嵌入指令微调
2. **YOLO微调**: 如检测效果不佳，可用合成的页面数据微调
3. **数据增强**: 增加更多草书变体数据

---

## 💡 提示

- 训练期间可随时按 `Ctrl+C` 中断，重新运行脚本会自动继续
- 最佳模型会自动保存到 `params/callialign_cursive_best.pth`
- 建议每天检查一次训练日志和GPU温度
- 如超过5天未完成，可考虑增加显卡

---

**开始时间**: 2026-03-23  
**预计完成**: 2026-03-26 (3天后)
