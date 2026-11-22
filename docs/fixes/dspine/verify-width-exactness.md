# 宽 verify 与逐比特一致性

投机解码的 verify 一次前向要提交 `k+1` 行（1 个 bonus + k 个草稿），而普通 decode 只有 1 行。
ggml 的 flash attention 按 `Q->ne[1]` 选 tile 几何和 KV 切分，MoE 的 `MUL_MAT_ID` 在 1 行与
4–16 行走不同的 batched 路径，LM head 在 ≥4 列时可能落进 tensor-core 的 MMQ 而非 MMVQ。因此
**同样一组 token，宽 verify 的数值结果与逐 token 解码并不保证逐比特相同**，累加顺序变了。

这意味着三件事，必须分开看：

1. **输出正确性**：投机接受的 token 仍由目标模型的 argmax/采样决定，被拒的草稿永远不进入输出。
   这一层是严格的，与宽度无关。
2. **逐比特可复现性**：同一个 prompt 在 `--draft-tokens 4` 与 `6` 下，贪心输出可能在第几百个
   token 处开始不同。这不是 bug，是 kernel 选择随宽度变了。要对比不同 k 的输出，只能用同一 k。
3. **基准可比性**：跨 k 比较 tok/s 是有效的（都是目标模型的真实解码），但"k=6 的输出更好/更差"
   这类结论不成立 —— 差异来自浮点累加顺序，不是模型质量。

## 我们的现状

- `--spec dflash|dflash2|mtp` 的 verify 宽度由 `--draft-tokens` 固定，或用 `--adaptive-mtp`
  让 `MaskedWidthController` 在 `K` 与 `K/2` 之间切换（每个宽度有自己捕获的 CUDA Graph）。
- 我们**没有**实现 alphallama 那种 batch-invariant 内核（`ALPHA_FA_BI_MAXQ`、MMVQ pinning、
  `ALPHA_SMALLM_TC`），所以无法让宽 verify 与 1 行 decode 逐比特相同。
- 实测：8,642 token 的 prompt，k=4 与 k=6 的接受率分别是 58.3% / 43.9%，decode 61.7 / 60.6
  tok/s。两者的第一个 token 相同，之后在高熵位置开始分叉。

## 实践建议

| 目的 | 做法 |
|---|---|
| 测吞吐 | 可以跨 k 比 tok/s，每组内部固定 k |
| 比输出质量 | **必须固定 k**，否则差异不可归因 |
| 复现问题 | 记录 `--draft-tokens` 与 `--adaptive-mtp` 状态；`--adaptive-mtp` 下宽度逐轮变化，输出天然不可复现 |
| 需要严格可复现 | 关掉投机（`--spec none`）或固定 k 且不开 `--adaptive-mtp` |

## 参考

alphallama 的 `apply_mtp_policy`（`alphainfer/src/engine/mtp_policy.cpp`）把这条约束做成了
显式模式：`--mtp-exact` 要求 verify 行与单行 decode 逐比特相同（代价是关掉小 M 的 MMQ 快路），
`--mtp-fast` 走快路但放弃该保证，并且在无法满足时**打印日志降级**而不是静默。
