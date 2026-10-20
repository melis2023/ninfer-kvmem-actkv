# KVMem action-region query：尝试与回退（2026-10-20）

## 1. 动机

ActKV（arXiv:2609.31395）的观察：agent 推理里 token 重要性不对称，只有 action 决定任务进度，用 observation/reasoning 的注意力偏好做 KV 淘汰会误删 action 需要的历史。

当前 KVMem 的 chat 打分 query 是"最后一条 real user message"。在 agent 循环里请求的最后一条消息通常是 `tool`，`real_user` 会跳过它，于是打分 query 落在很多轮之前的初始提问上。目标是把打分 query 移到"基于最新 observation 决定下一步 action"的上下文。

## 2. 第一次尝试：直接覆盖 memory_query

在 `frontend.cpp`（graft 之后）：

```cpp
if (result.memory_query && !message_roles.empty() &&
    message_roles.back() == ChatRole::Tool) {
    constexpr std::size_t action_query_tokens = 4096;
    const std::size_t end = result.token_ids.size();
    const std::size_t begin = end > action_query_tokens ? end - action_query_tokens : 0;
    result.memory_query = TokenSpan{begin, end - begin};
}
```

结果（256K NIAH）：

| 轮次 | probe 耗时 |
|---|---|
| round 1–3 | 1.5 / 1.7 / 1.6 s |
| round 4 | 89.2 s |
| round 5–11 | 29.8 – 32.2 s |
| round 12–15 | 275.3 / 294.1 / 310.4 / 328.2 s |
| final sweep batch | 330.0 / 329.7 s |

召回没有崩（到 batch 2 仍是全中），但速度不可接受。

### 根因

`memory_query` 同时承担两个职责：

1. **复用锚点**：`memory_checkpoint_frontier` 返回 `window.query_begin`，`memory_same_query` 要求
   `query.begin == window.query_begin`（`kvmem_window.cpp:930`）。
2. **打分范围**：`prepare_memory_statistics` 用它设置 GPU 累积窗口。

tool 轮的 `end` 每轮都变，于是锚点每轮漂移，历史 checkpoint 全部命中 `query_rejected`
（`kvmem_window.cpp:1092`），每轮 probe 都退化成重新 prefill 整个 250K 上下文。

## 3. 第二次尝试：拆出 score_query

改动面：

| 文件 | 内容 |
|---|---|
| `prepared_prompt.h` | 新增 `std::optional<TokenSpan> score_query` |
| `frontend.cpp` | tool 轮只设 `score_query`，`memory_query` 保持稳定 |
| `program_impl.h` | `KvmemWindowState` 新增 `score_begin/score_end/score_tokens`；`prepare_memory_statistics` 增加两个参数 |
| `memory_statistics.cpp` | GPU 累积窗口用 score 范围；`query_tokens` 与 `score_tokens` 分开累加 |
| `prefill.cpp` / `kvmem_window.cpp` | 打分处改用 `score_tokens` |

这个拆分本身是正确的：锚点稳住，复用恢复。但暴露出第二个耦合——

`commit_memory_statistics` 只在 `!state.query_replayed` 时累积 `query_sum`：

```cpp
if (!state.query_replayed && query_last > query_first) { ... state.query_sum[i] += query[i]; ... }
```

tool 轮会命中 checkpoint 复用，`query_replayed == true`，于是**新 observation 的分数根本不会进入
`query_sum`**，打分仍然用上一轮的旧 query。想让 action-region 真正生效，必须把打分改成增量累积
（只累积本次新增的 token 段），并且让 `memory_checkpoint_frontier` 覆盖 score 窗口。这是独立的一次改动，
不是参数级调整。

## 4. 当前状态

第 3 点**已回退**。回退后源码等价于已通过验证的 LRFU EMA 版本：

- `kvmem/include/kvmem/lrfu.hpp`：指数加权平均
- `backends/ninfer/.../frontend/frontend.cpp`：token-only prompt 尾部 1024 窗口
- 256K NIAH：33/33 strict、33/33 union、14/14 cross-tool
- 64K 回归：prefill 435.3 tok/s、decode 49.2 tok/s、draft 接受率 51.3%

已回退的文件（`git checkout`）：

```
src/models/qwen3_5/frontend/prepared_prompt.h
src/models/qwen3_5/program/program_impl.h
src/models/qwen3_5/program/prefill.cpp
src/models/qwen3_5/program/storage/memory_statistics.cpp
src/models/qwen3_5/program/storage/kvmem_window.cpp
```

`frontend.cpp` 只保留 token-only 的 1024 尾部窗口。

## 5. 正确的实现路径（后续）

1. 保留 `score_query` 拆分（第 3 节的改动）。
2. 把 `query_sum` 的累积从"一次性、复用即跳过"改为"每次 prefill 累积本次新增段"：
   移除 `!state.query_replayed` 这个前置条件，改为按 `[max(score_begin, begin), min(score_end, end))` 累积。
3. 让 `memory_checkpoint_frontier` 在 score 窗口存在时返回 `min(query_begin, score_begin)`，
   并在该 frontier 捕获/恢复，保证 score 窗口完整。
4. 代价评估：tool 轮需要 replay `end - score_begin` 个 token（上限 4096），约 10 s 量级；
   需要先确认这不会抵消 ActKV 的收益。

在完成第 2、3 步之前，不建议单独上线 action-region 打分。
