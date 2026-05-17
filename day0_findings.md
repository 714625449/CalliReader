# Day 0 Findings

## Probe 3: Original 正楷/草书区分方式
- 检查 CaoshuMerged/Training 下多个字类目录的文件名
- 文件名模式：纯数字 ID（如 131037.jpg, 0.jpg, 3.jpg），无书体标记
- 无 _kaishu_ / _caoshu_ / 草书 / 正楷 等关键字
- 搜索全局元数据：无 CSV/JSON/TXT 标签文件
- **结论：无法自动区分正楷/草书 → 放弃辅助长尾集，只保留主验证集 A + B**

## Probe 4: 整幅书法图来源
- `/caoshu/imgs/`：约 20 张整图
- `/caoshu/imgs/samples/samples/images/`：**1,000 张整幅书法图**，尺寸 776×2000 ~ 2000×1360，RGB
- **结论：端到端验证集来源充足，可直接构建主验证集 B**

## Probe 1: Paired Control (shufazidian vs CCC/Val original)
- **状态：运行中**（PID 501132，60K 张图推理，预计 25-30 分钟）
- 待完成后再追加结果
