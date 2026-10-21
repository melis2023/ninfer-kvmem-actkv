# Layer-wise 打权实验（2026-10-22，已回退）

## 0. 结论

**无收益，已回退。**

- 128K：20/20，与 ActKV 基线持平
- 192K：18/20，与基线持平，失败项完全相同
- 240K：**15/20，比基线少 1 题**（`pos2-detail-4` 从 OK 变 refusal）
- 打分开销：64K 单次 31.6 ms → 42.2 ms（+34%）
- 端到端：240K 450.4 s vs 基线 444.7 s（+1.3%，同参数对比）

动机是 PyramidKV 的观察：浅层 full-attention 偏局部、深层偏语义，统一 top-K 是简化。但本模型（Qwen3.8-27B，16 个 full-attention 层）上这个假设没有兑现——层权重只改变了被选中块的集合，没有改善深层召回，反而在 240K 多引入一次拒答。

## 1. 改动内容

`kvmem/src/host/mean_key_retrieval.cpp`：

```cpp
// 跨层加权平均，权重从第一层 1.3 线性降到最后一层 0.8
std::vector<double> layer_weight(query.layers, 1.0);
if (query.layers > 1) {
    for (uint32_t layer = 0; layer < query.layers; ++layer)
        layer_weight[layer] = 1.3 - 0.5 * double(layer) / double(query.layers - 1);
}
...
total += w * layer_score / query.kv_heads;
weight_sum += w;
...
scores[block] = weight_sum > 0.0 ? total / weight_sum : 0.0;
```

原实现是 `total / active_layers`（等权）。改动只涉及 host 打分，不动内核、不动选块数（仍 576）、不动 B/R 配比。

单测 `kvmem/tests/mean_key_retrieval_test.cpp` 的 FP64 oracle 同步改为按同一权重定义（`1.3 → 0.8`，除以 active 层权重和），否则旧 oracle 的等权期望必然失败。

## 2. 结果

### 2.1 64K 单请求（`workers/ninfer/sweep.json`）

| 项 | ActKV 基线 | layer-wise | 差异 |
|---|---:|---:|---:|
| prefill tok/s | 443.4 | 428.2 | −3.4% |
| decode tok/s | 46.7 | 44.5 | −4.7% |
| MTP 接受率 | 48.1% | 47.5% | −0.6 pt |
| elapsed | 138.2 s | 143.2 s | +5.0 s |

单次 64K 的 run-to-run 波动本身约 ±3%（同一 exe 曾测到 404.7 与 443.4），此项不足以定性。

### 2.2 256K NIAH（33 针，15 轮 × 435 filler）

| 项 | ActKV 基线 | layer-wise |
|---|---:|---:|
| same-turn | 33/33 | 33/33 |
| final strict | 33/33 | 33/33 |
| final union | 33/33 | 33/33 |
| cross-tool | 14/14 | 14/14 |
| 工具调用 | 80 | 80 |
| final sweep prompt | 398 | 398 |
| write 合计 | 564.9 s | 572.3 s |
| probe 合计 | 21.4 s | 21.7 s |
| final sweep | 9.5 s | 10.4 s |
| 端到端 | 595.8 s | 604.4 s |

召回零变化，端到端 +1.4%（15 轮平均，可信）。

### 2.3 LongMemEval（20 题/档，idxs 0,1,2）

| 配置 | 128K | 192K | 240K | 合计 |
|---|---:|---:|---:|---:|
| EMA | 19/20 | 16/20 | 15/20 | 50/60 |
| ActKV LRFU | 20/20 | 18/20 | 16/20 | 54/60 |
| **layer-wise** | **20/20** | **18/20** | **15/20** | **53/60** |

192K 失败项完全一致（`deep-original`、`mid-detail-7`），只是 `deep-original` 从 miss 变 refusal。

240K 失败项对比：

| 项 | ActKV | layer-wise |
|---|---|---|
| deep-original | miss | miss |
| mid-original | miss | miss |
| mid-detail-3 | miss | miss |
| pos2-detail-4 | OK | **refusal（新增）** |
| pos2-detail-5 | refusal | refusal |

240K 耗时（参数已对齐 `--write-tokens 1 --max-server-tokens 240000`，final prompt 均为 242,370）：**450.4 s vs 444.7 s，+5.7 s（+1.3%）**。

## 3. 打分开销测量（本次留下的有效数据）

`KVMEM_RESELECT` 日志（256K NIAH，blocks > 1000 段）：

| 项 | 数值 |
|---|---:|
| 重选次数 | 85 |
| 有效打分次数（query 窗口完整） | 62 |
| 打分总计 | 10.07 s |
| 单次均值 | 162.4 ms |
| 单次范围 | 57.9 – 241 ms |
| 搬页总计 | 16.52 s |

同一份日志的 64K 段（blocks ≤ 1000）：

| 项 | 基线 | layer-wise |
|---|---:|---:|
| 打分均值 | 31.6 ms | 42.2 ms |
| 选块均值 | 25 µs | 36 µs |
| 搬页均值 | 129 ms | 159 ms |

结论：**打分占 256K 端到端 1.7%（10.07 s / 604.4 s）**，是 host 端唯一可并行的部分。块间无依赖，可做 block 级并行：

| 线程 | 打分剩余 | 节省 | 占端到端 |
|---:|---:|---:|---:|
| 4 | 2.52 s | 7.6 s | 1.2% |
| 6 | 1.68 s | 8.4 s | 1.4% |
| 8 | 1.26 s | 8.8 s | 1.5% |

上限被 Amdahl 锁死：打分只占 1.7%，无论如何并行都省不出更多。搬页（16.52 s）更大但受 PCIe 限制，并行无效。

## 4. 为什么层权重没起作用

三个可能原因，按可信度排序：

1. **选块数固定为 576，层权重只改变排序，不改变总量。** 576/4000 ≈ 14% 的保留率下，浅层加权让浅层块多进 1–2 个、深层少进 1–2 个，这个摆动小于 LRFU 的分数衰减幅度，排序几乎不变。
2. **16 个 full-attention 层不是均匀分布的。** Qwen3.5 的 full-attention 层嵌在 64 层里，层索引 0–15 不等于深度位置；线性 1.3→0.8 的映射与真实深度不对应。
3. **mean key 本身是低通滤波。** 层内的信息在取均值时已经损失，跨层再加权是在已经模糊的信号上做二阶调整。

## 5. 回退

```bash
git checkout -- kvmem/src/host/mean_key_retrieval.cpp kvmem/tests/mean_key_retrieval_test.cpp
```

回退后重新编译，`mean_key_retrieval_test`、`kvmem_store_test` 均通过；64K 恢复 424.7 / 43.5 / 48.1%。

## 6. 后续可选

| 方向 | 预期 | 风险 |
|---|---|---|
| block 级并行打分 | 256K 省 7–9 s（1.2–1.5%） | 低，纯 host，`std::thread` 分块 |
| 增量打分（只算新增块） | 打分降 90% | 中，需处理 query 变化 |
| 继续调 B/R | 未知 | 零代码 |

前两项可以叠加；并行打分是确定性收益，但幅度只有 1.5%。
