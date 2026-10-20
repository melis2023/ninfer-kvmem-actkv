# Per-token 打分：实现与失败根因（2026-10-20）

## 1. 目标

把块级打分（64-token 均值 key 的余弦）升级成 token 级：

```
score_t = dot(k_t, q_mean) / ||k_t||
block_score = aggregate(score_t)
```

如果可行，同窗口可以保留 3–5 倍有效 token，为后续 compaction 铺路。

## 2. 实现

| 层 | 改动 | 文件 |
|---|---|---|
| GPU kernel | 每 token 一个标量，GQA 分组平均 query | `ops/token_sums/token_sums.cu` |
| 布局 | `token_scores[capacity, layers]` FP32，16 MB | `execution/memory_statistics.h` |
| 分配 | `startup.cpp` / `program_impl.cpp` | 同左 |
| 写入 | prefill 时每层写一列 | `execution/attention.cpp` |
| 读回 | `cudaMemcpy2DAsync` 按层拉回 host | `memory_statistics.cpp` |
| 聚合 | 块内 max + z-score + tanh | `kvmem_window.cpp` |
| 融合 | `0.7 · mean_key + 0.3 · token` | 同左 |
| 开关 | `NINFER_TOKEN_SCORES=1` | 同左 |

## 3. 两次失败

### 3.1 纯 token 分数（替换 mean-key）

| 指标 | 结果 |
|---|---:|
| same-turn | 崩（round 6 起 0/2） |
| final | 4/14 cross-tool |
| 64K decode | 43.6 tok/s（基线 49.1） |
| 接受率 | 43.2%（基线 51.3） |

### 3.2 融合 30%（块内 max + z-score + tanh）

| 指标 | 结果 |
|---|---:|
| same-turn | **22/33** |
| final strict | **29/33** |
| union | 29/33 |
| cross-tool | 13/14 |
| 64K prefill | 435.6 tok/s |
| 64K decode | 45.6 tok/s |
| 接受率 | 44.1% |

失败模式一致：**round 6 开始崩**，正好是上下文超过 42.6K、触发第一次重选的时刻；round 14–15 又部分恢复。

## 4. 根因：流式 prefill 与完整 query 的时序矛盾

`token_scores` 在 prefill 的每个 chunk 写入，用的是**当时已经累加的 `q_sum`**：

```
chunk 0:  q_sum = sum(query[0..256))      → 写 token_scores[0..256)
chunk 1:  q_sum = sum(query[0..512))      → 写 token_scores[256..512)
...
chunk N:  q_sum = sum(query[0..query_end)) → 写 token_scores[...]
```

问题：

1. `q_sum` 是 query 区域的**累积和**，前期 chunk 看到的是不完整的 query；
2. 早期 token 的分数用「半截 query」算出来，方向和量级都偏；
3. 后期 token 用完整 query；
4. **同一次选择里，不同块的 token 分数尺度不一致**，块内 max 或均值都不可比。

这就是为什么：

- 短上下文（round 1–5）`q_sum` 很快完整，分数一致，召回正常；
- 长上下文（round 6 起）前期 token 的分数严重失真，选择崩。

块级 mean-key 没有这个问题：`mean_key_retrieval_scores` 在 prefill 结束后用**完整的 `query_sum`** 对块均值做一次余弦，所有块用同一个 query。

## 5. 为什么不能简单修

| 方案 | 代价 | 结论 |
|---|---|---|
| 两遍 prefill（第一遍算 query，第二遍算 token 分数） | prefill 时间 ×2 | 不可接受 |
| 存全量 per-token key 供事后重算 | 17 GB 显存 | 不可行 |
| 只在最后一个 chunk 写 token 分数 | 只覆盖尾部 256 token | 无效 |
| 用块级 mean key 反推 token 分数 | 信息不足 | 回到现状 |
| 让 `q_sum` 在 prefill 前就完整（预扫 query） | 多一次 query prefill | 等价于两遍 |

**根本约束**：token 级打分要求「完整 query 向量」在写 token 分数时已经就绪，而流式 prefill 是「边读边写」的。这两个要求互斥。

## 6. 当前状态

- 默认关闭：不设 `NINFER_TOKEN_SCORES` 时走块级 mean-key，召回 33/33；
- kernel 保留，设 `NINFER_TOKEN_SCORES=1` 可复现失败，用于后续研究；
- 64K 基线恢复：prefill 422.9、decode 49.1、接受率 51.3%。

## 7. 如果以后要重做

必须先解决 query 完整性问题，可选路径：

1. **预扫 query**：prefill 前先单独算一遍 query 区域的 hidden 均值（约 query 长度的 prefill 成本），再开始正式 prefill；
2. **延迟写入**：token 分数先写进临时缓冲，prefill 结束后用完整 query 统一重算一遍（需要保留 normalized_key 的 chunk 副本）；
3. **训练侧方案**：像 EdgeRock 的 RMA 那样，让模型学会写时不依赖 query（需要训练）。

三条都不是「改几行」的量级。

## 8. 相关

- `docs/kvmem-actkv-lrfu-20261020.md`
- `docs/kvmem-incremental-restore-20261020.md`
- ActKV arXiv 2609.31395 §5.1（action attention）
- EdgeRock arXiv 2610.10118 §3.1（learned writer）
