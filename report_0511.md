# 草书识别项目进展报告（2026-05-11）

## 1. 项目概述

基于 InternVL 架构的草书（Cursive Chinese Calligraphy）识别系统，采用分离式 Pipeline：

```
图片 → YOLO分割 → 单字识别(Resampler) → Top-K匹配 → 后处理拼接
```

## 2. 组件状态

| 组件 | 状态 | 权重文件 | 说明 |
|------|------|----------|------|
| YOLO分割 | ✅ | `params/best.pt` | 分割单字，可用 |
| Vision Model | ✅ | `params/vit_model.pt` | 提取视觉特征，冻结 |
| MLP1 | ✅ | `params/mlp1.pth` | 特征降维，冻结 |
| **Resampler** | 🔄 | `params/callialign.pth` | **核心瓶颈，正在重训** |
| Token Embeddings | ✅ | `params/token_embedding.pth` | 字符embedding，冻结 |
| e-IT LoRA | ⚠️ | `outputs/eit_simple_overfit/final` | 辅助纠错，非主力 |

## 3. 训练结果

### 3.1 Resampler（当前主力）

| 阶段 | 训练样本 | 验证样本 | Top-1 | Top-5 | 说明 |
|------|---------|---------|-------|-------|------|
| 初训 (callialign) | 6,493 | 4,281 | **29%** | ~50% | 数据不足，严重欠拟合 |
| **重训中 (v2)** | **974,113** | **61,181** | **23.30%** @5k | 39.05% | 验证集更严格，早期阶段 |

> 重训进度：step 5000+/100000，best loss 0.4658，磁盘可用 50GB

### 3.2 e-IT LoRA 微调（已验证效果有限）

- 539 样本训 10 epoch，Loss 0.059
- **端到端测试失败**：Visual 路径不兼容，知识无法迁移
- **重复问题改善**：基线疯狂重复，LoRA 输出格式正常
- **结论**：不用于端到端读图，仅作后处理纠错备用

## 4. 关键教训

1. **数据量决定上限**：6K 样本 → 29%，974K 样本 → 目标 40%+
2. **Visual 路径必须一致**：训练和推理用不同 Resampler = 白训
3. **Loss 低不等于效果好**：e-IT Loss 0.059 但实际幻觉严重
4. **验证集规模影响指标**：大验证集（61K）比小验证集（4K）更真实
5. **磁盘管理**：每个 ckpt 3.2GB，需自动清理机制

## 5. 当前策略

```
┌──────────────────────────────────────────┐
│  主攻：Resampler 大数据量训练             │
│  辅助：e-IT LoRA 后处理纠错（非优先）     │
│  不搞：端到端 InternVL + LoRA（已证伪）   │
└──────────────────────────────────────────┘
```

## 6. 下一步

- [ ] Resampler 继续训练至 100,000 步
- [ ] 观察 Validation Top-1 是否突破 30%、35%、40%
- [ ] 如效果达标，用新 Resampler 替换 callialign.pth 跑完整 Pipeline
- [ ] 探索 LLM 后处理：Top-3 → 语义纠错

## 7. 文件索引

| 文件 | 用途 |
|------|------|
| `TRAINING_SUMMARY.md` | 完整训练总结（详细版） |
| `log0510.md` | 逐日训练日志 |
| `caoshu/train.py` | Resampler 训练脚本 |
| `caoshu/pipeline.py` | 整图识别 Pipeline |
| `scripts/resume_resampler_train.sh` | 当前训练启动脚本 |
| `scripts/merge_datasets.py` | 数据集合并工具 |
| `scripts/e2e_image_test.py` | 端到端测试脚本 |

---

*报告时间：2026-05-11*  
*训练状态：Resampler 恢复训练中（step 5000+/100000）*  
*磁盘状态：/root/sj-tmp 可用 50GB*
