# KVMem action-region query：尝试与结论（2026-10-20）

## 1. 动机

ActKV（arXiv:2609.31395）的观察：agent 推理里 token 重要性不对称，只有 action 决定任务进度，用 observation/reasoning 的注意力偏好做 KV 淘汰会误删 action 需要的历史。

当前 KVMem 的 chat 打分 query 是"最后一条 real user message"。agent 循环里请求的最后一条消息通常是 `tool`，`real_user` 跳过它，打分 query 落在很多轮之前的初始提问上。目标是把打分 query 移到"基于最新 observation 决定下一步 action"的上下文。

## 2. 尝试一：直接覆盖 memory_query

```cpp
if (result.memory_query && !message_roles.empty() &&
    message_roles.back() == ChatRole::Tool) {
    result.memory_query = TokenSpan{end - 4096, 4096};
}
```

256K NIAH 结果：

| 轮次 | probe 耗时 |
|---|---|
| 1–3 | 1.5 / 1.7 / 1.6 s |
| 4 | 89.2 s |
| 5–11 | 29.8 – 32.2 s |
| 12–15 | 275.3 – 328.2 s |
| final sweep | 330.0 s |

### 根因

`memory_query` 同时是**复用锚点**和**打分范围**。`memory_checkpoint_frontier` 返回 `window.query_begin`，`memory_same_query` 要求 `query.begin == window.query_begin`。tool 轮末尾每轮都变，锚点漂移，历史 checkpoint 全部 `query_rejected`，每轮 probe 退化成重新 prefill 250K。

## 3. 尝试二：拆出 score_query

新增 `PreparedPromptData::score_query`、`KvmemWindowState::score_begin/score_end/score_tokens`，`prepare_memory_statistics` 增加两个参数，GPU 累积窗口改用 score 范围，`memory_query` 保持稳定。

这个拆分方向正确：锚点稳住，复用恢复。但暴露出第二个耦合：

`commit_memory_statistics` 只在 `!query_replayed` 时累积 `query_sum`。tool 轮命中 checkpoint 复用时 `query_replayed == true`，新 observation 的分数根本进不了打分。

### 改成增量累积

移除 `!query_replayed` 条件，改为按 `[max(score_begin, statistics_frontier), score_end)` 累积。结果 64K 的 `query_tokens` 变成 `118457`，正好是 `query_end=59327` 的两倍。

根因：`rewind_memory_query` 的 query replay 会把 `[query_begin, query_end)` 整个重跑一遍，首次 prefill 已经累积过一次。修法是 replay 前清空累积：

```cpp
window.query_replayed = true;
window.query_tokens = 0;
window.score_tokens = 0;
std::fill(window.query_sum.begin(), window.query_sum.end(), 0.0F);
restore_memory_prefix(sequence, checkpoint, false, true);
```

64K 恢复：`query_tokens=59133`，最后一次重选 `score_us=42629`。

## 4. 尝试三：256K 召回

| 指标 | 回退版（LRFU EMA） | 增量打分 |
|---|---:|---:|
| same-turn | 33/33 | 33/33 |
| final strict | 33/33 | **4/33** |
| final union | 33/33 | **4/33** |
| cross-tool | 14/14 | **0/14** |
| 重选次数 | ~120 | 121 |
| 有打分的重选 | 全部 | **0** |

`score_us=0` 的原因：`query_complete` 要求 `begin >= score_end`，而重选发生在 prefill 途中（`logical=257273` < prompt 258K），评分窗口在 prompt 末尾，永远等不到完整。

## 5. 尝试四：部分打分

把条件放宽为 `score_tokens != 0`，用截至重选点已累积的部分打分。

结果：`score_gt0=51`（确实打了 51 次分），`selected=576`，但召回仍然：

| 指标 | 结果 |
|---|---:|
| same-turn | 33/33 |
| final strict | **0/33** |
| final union（1 次重试后） | **4/33** |
| cross-tool | **0/14** |

只有 N30–N33（最后两轮）被保留。

### 根因

NIAH 每轮 write 的 `memory_query` 是**最新 16K 针列表**，不是全部针。增量打分只累积 `[statistics_frontier, query_end)` 这一段，也就是最新一批针。旧针不在评分窗口内，每轮 hit=0，LRFU EMA（λ=0.5，约 2 轮记忆）两个轮次后就把它们衰减到 0。

回退版之所以是 33/33，是因为它的 `query_sum` 在复用时**不重新累积**（`!query_replayed` 跳过），保持了上一次完整窗口的分数，EMA 有稳定的输入。

## 6. 结论

在现有 KVMem 架构下，action-region 打分不可行，原因是一个三角约束：

1. **复用锚点必须稳定** → `memory_query` 不能随 tool 轮移动；
2. **打分必须完整** → 重选时评分窗口要覆盖到重选点，否则只能用部分前缀；
3. **每轮 query 会变** → 旧的 `query_sum` 不能直接复用。

要同时满足这三点，只有三条路，每条都要改现有契约：

- **重选等待评分窗口完成**：把压力重选推迟到 query 累积完之后。会拉长 prefill 峰值延迟，且窗口在 prompt 末尾时等于永不重选。
- **跨轮保留旧分数**：给 LRFU 加"非评分轮不衰减"的语义。但这样旧针分数不再反映当前相关性，等于放弃 LRFU 的时间衰减。
- **每轮全量重评分**：放弃前缀复用，回到 250K 全量 prefill。写入 43 s → 650 s，不可接受。

## 7. 当前状态

已回退到 `51b7dd8c`，工作区干净。保留的改动只有：

- `kvmem/include/kvmem/lrfu.hpp`：指数加权平均
- `kvmem/tests/kvmem_store_test.cpp`：EMA 断言
- `backends/ninfer/.../frontend/frontend.cpp`：token-only prompt 尾部 1024 窗口

`backends/ninfer` 已回退到 `51b7dd8c`，与主仓库 `82118ad` 记录的指针一致。

回退后 64K：prefill 430.5 tok/s、decode 48.3 tok/s、draft 接受率 51.3%。

## 8. 如果以后还要做

先做一个不改变打分时序的实验：**只在 write 轮（最后一条是 user）用尾部窗口，tool 轮不动**。这样 query 不会每轮漂移，增量累积只发生在 write 轮，`query_sum` 的语义不变。代价是 agent 的 action 轮仍然用旧的 user query，ActKV 收益有限，但不会破坏召回。

真正要让 action 轮生效，建议先给 `KvmemWindowState` 增加一个"本轮 score 窗口是否已完整"的字段，并让 `memory_pressure` 在窗口未完整时延迟重选一个 chunk，观察对 prefill 峰值的影响，再决定是否值得。
