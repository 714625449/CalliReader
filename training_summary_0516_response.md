# 对 Contrastive Loss 行动计划的建议

> 日期：2026-05-16

---

## 1. 分开记录 pos_loss 和 neg_loss（强烈建议）

现在函数只返回 total_loss。如果 batch 内没 hard negative（大概率），neg_loss ≈ 0，你看 log 根本分不清 contrastive 项有没有在起作用。

改成返回或打印三个值：total_loss、pos_loss、neg_loss。跑 2-3K 步后如果 neg_loss 一直 < 0.001，说明 margin 太高或随机 batch 里没有 hard negative，需要立刻调整。

## 2. margin=0.1 可能偏高

高维归一化空间中随机向量余弦相似度期望 ≈ 0。即使是形近字，它们的文本 embedding 相似度未必 > 0.1。

建议先离线跑一次：从 `all_char_embeds_norm` 算相似度分布，看最相似的字对 cosine sim 是多少。如果大部分 < 0.1，那 `F.relu(sim - 0.1)` 永远是 0，contrastive 项形同虚设。

备选方案：

- `margin=0.0`（惩罚任何正相似度）
- 或者换成 **InfoNCE**（不需要 margin，用 softmax 自动关注最难负样本）：

```python
# InfoNCE 替代方案
logits = sim_matrix / temperature  # temperature=0.07
labels_nce = torch.arange(B, device=device)
neg_loss = F.cross_entropy(logits, labels_nce)
```

## 3. Optimizer state

从 99K resume 会加载 Adam 的 momentum/variance，这些是针对旧 loss 估计的。新 contrastive 项改变了梯度方向。

通常不是大问题（pos_loss 占主导），但如果前 1-2K 步 val 剧烈下跌且不恢复，可以考虑只加载 model state、不加载 optimizer state。

## 4. total_steps 建议改 140K

21K 步够看趋势，但如果有效可能刚起势就被截断。设 140K 更保险，反正随时能停。

---

## 优先级总结

| 建议 | 紧急度 | 改动量 |
|------|--------|--------|
| 分开 log pos/neg loss | 高 | 5 行代码 |
| 离线检查 margin 合理性 | 高 | 跑一次脚本 |
| 考虑 InfoNCE 替代 | 中 | 备选 |
| total_steps 改 140K | 低 | 一个参数 |

---

## 核心观点

**先验证 contrastive 项确实在产生梯度**，再跑长训练。否则可能白跑 20K 步才发现 neg_loss 一直是 0。
