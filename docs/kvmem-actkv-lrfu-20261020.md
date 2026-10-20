# KVMem 打分：ActKV LRFU 公式（2026-10-20）

## 1. 背景

`kvmem-lrfu-ema-20261020.md` 记录的 EMA 公式已被本文件取代：

```cpp
// EMA（旧）
KS_t = λ · KS_{t-1} + (1-λ) · score    // hit
KS_t = λ · KS_{t-1}                    // miss

// ActKV LRFU（现行）
KS_t = λ · KS_{t-1} + score            // hit，加原始分数
KS_t = λ · KS_{t-1}                    // miss，只衰减
```

新块按 ActKV Algorithm 1 的 `Init(L_ora, 0)` 从 0 开始，第一次 hit 时被当前分数点亮。

## 2. 依据

ActKV（arXiv 2609.31395）Algorithm 1：

```
M_hit  ← GetHitMask(AttnScore, p)          // p = top 90%
KS[M_hit]  ← λ · KS[M_hit] + AttnScore[M_hit]
KS[M_miss] ← λ · KS[M_miss]
KeepMask, KS' ← TopK(KS, k = B)
```

论文的 AAV（action access vector）分析把条目分成三类：

| 类别 | 含义 | 论文结论 |
|---|---|---|
| cold-start | 开头有 0，后期才重要 | 可忽略 |
| **intermittent** | 中间有 0，不连续命中 | **每次访问贡献最高** |
| continuous | 无 0，连续命中 | 常见但单次贡献低 |

`λ·KS + score` 让间歇性高分条目在长时间不命中后仍能存活，而 `λ·KS` 的衰减让过时的连续条目被淘汰。EMA 的 `(1-λ)·score` 会把间歇条目的单次高分削掉一半，保护不足。

## 3. 不改内核的边界

论文的打分输入是 action-region attention：

```
AttnScore ← CalAttn(Q_action, K_cache)
```

我们拿不到 attention（paged KV 后端不暴露，标准 kernel 需要连续内存）。当前输入仍是 `mean_key_retrieval_scores` 的 block mean key 余弦：

```
query_sum   = memory_query 区域 hidden 均值
key_means   = 每个 64-token 块的 K 均值
score       = cosine(query_sum, key_means)
```

所以这次改动只落地了论文的**时间维度**（LRFU 更新规则），没有落地**空间维度**（action attention）。要拿 action attention，需要 kernel 输出 attention 或 LSE。

## 4. 测试

### 4.1 单元测试

`kvmem_store_test` 通过。第二轮断言按新公式更新：

```cpp
// λ=0.5: 0.5*0.8 + 0.1 = 0.5, 0.5*0.6 + 0.4 = 0.7
CHECK(std::abs(store.selection_score(2) - 0.5) < 1e-12);
CHECK(std::abs(store.selection_score(3) - 0.7) < 1e-12);
```

### 4.2 64K 回归

| 指标 | EMA | ActKV LRFU |
|---|---:|---:|
| prefill | 422.4 tok/s | 399.7 tok/s |
| decode | 49.3 tok/s | 46.3 tok/s |
| draft 接受率 | 51.3% | 51.3% |
| 增量 restore | 124–181 页 | 124–181 页 |

### 4.3 256K NIAH 33 针

| 指标 | EMA | ActKV LRFU |
|---|---:|---:|
| same-turn | 33/33 | **33/33** |
| final strict | 33/33 | **33/33** |
| final union | 33/33 | **33/33** |
| cross-tool | 14/14 | **14/14** |
| 工具调用 | 80 | **80** |
| 端到端 | 594.8 s | 646.9 s |
| write prefill | 368–976 | 358.8–925.2 |
| probe decode | 57.9–81.2 | 57.3–77.2 |

召回零差异。端到端慢 8.8%，但 write prefill 的波动区间与历史几次（588–637 s）重叠，不能单独归因。

NIAH 区分不出两种公式，因为它的针一旦写入后每轮都被同一 query 命中，属于 continuous 条目。

### 4.4 Agent 循环 220K（`agent_256k_toolrecall.py`）

30 轮，每轮 `search_archive → fetch_document → record_finding`，每轮 100 条档案结果。

| 指标 | 结果 |
|---|---:|
| 逐轮工具召回 | **30/30** |
| final 召回 | **30/30** |
| 工具调用 | 181 |
| findings | 60 |
| 最大上下文 | 219,948 |
| 总时间 | 341.3 s |
| 重选 | 43 |
| 增量 restore | 33/187（17.6%） |

上下文从 182K 增到 220K，时间只增加 1.9%（335.1 → 341.3 s），因为 prefix cache 命中 99.9%，窗口固定 53K。

## 5. 结论

- 召回：NIAH 33/33、agent 30/30，与 EMA 一致；
- 速度：NIAH 慢 8.8%，agent 场景未对比 EMA；
- 公式更贴近 ActKV，且纯 host，保留成本低；
- 真正的收益要等 action-region attention（需要 kernel），当前仅落地时间维度。

## 6. 回退

```bash
git checkout -- kvmem/include/kvmem/lrfu.hpp kvmem/tests/kvmem_store_test.cpp
```

或使用备份
`I:\llama\_backup\kvmem-llama.cpp-multi-backend-framework-20261020-143834`（该备份是 EMA 版本）。

## 7. 相关文档

- `docs/kvmem-lrfu-ema-20261020.md`：EMA 版本的动机与数据（历史记录）
- `docs/kvmem-incremental-restore-20261020.md`：差异 restore
- `docs/kvmem-actkv-query-20261020.md`：action-region query 的失败尝试
