# KVMem 检索打分：LRFU 归一化与 query 尾部窗口（2026-10-20）

## 1. 动机

### 1.1 LRFU 的年龄偏差

当前 `LrfuHistory` 的 keep score 是一个无界累积和：

```
KS_t(i) = lambda * KS_{t-1}(i) + s_t(i) * [s_t(i) >= top_fraction 阈值]
```

`top_fraction` 默认 0.9，意味着约 90% 的 block 每轮都算 hit。于是 KS 接近"历史分数总和"：一个早期高分、后来不再相关的 block，可以仅凭"存在得久"压过当前真正相关的 block。lambda=0.5 时 8 轮后旧分数衰减到 1/256，偏差有限，但在长对话里仍然存在，而且没有任何机制把"平均质量"和"累积总量"区分开。

InfiLoop（arXiv:2610.11570）的做法是维护 `N_t = beta*N_{t-1} + e_t*z_t` 和 `D_t = beta*D_{t-1} + e_t`，用 `N_t/D_t` 作为状态：归一化后的量是**加权平均**，不随迭代次数增长。

### 1.2 token-only prompt 的 query 稀释

`Frontend::prepare_tokens` 把所有 token 化 prompt 的 `memory_query` 设成 `{0, token_ids.size()}`。`mean_key_retrieval_scores` 用的是 query 的均值向量，因此 247K token 的 raw prompt 会得到一个"全 prompt 平均"的 query，和任何单个 block 的余弦相似度都挤在很窄的区间里，排序信息量极低。

chat 请求不受影响：`chat_template.cpp` 把 `memory_query` 设成最后一条 real user message 的内容区域。

## 2. 改动

### 2.1 LRFU 从累积和改为指数加权平均

`kvmem/include/kvmem/lrfu.hpp`

```cpp
for (size_t i = 0; i < scores.size(); ++i) {
    const double hit = scores[i] >= threshold ? scores[i] : 0.0;
    (*next)[i] = i < state_.scores->size()
        ? lambda_ * (*state_.scores)[i] + (1.0 - lambda_) * hit
        : hit;
}
```

- 已见过的 block：`KS_t = lambda*KS_{t-1} + (1-lambda)*hit`，稳态范围与检索分数一致。
- 第一次出现的 block：直接用当前分数初始化，不再因为"没有历史"被系统性低估。
- 快照格式不变（仍然只存 `scores`），不需要迁移。
- `lambda` 的语义从"累积衰减"变成"平均窗口长度"：lambda=0.5 约等于最近 2 轮，lambda=0.8 约等于最近 5 轮。

### 2.2 token-only prompt 只对尾部打分

`backends/ninfer/src/models/qwen3_5/frontend/frontend.cpp`

```cpp
constexpr std::size_t token_query_tail = 1024;
const std::size_t query_begin = result.token_ids.size() > token_query_tail
    ? result.token_ids.size() - token_query_tail : 0;
result.memory_query = TokenSpan{query_begin, result.token_ids.size() - query_begin};
```

- 短 prompt（<= 1024 tokens）保持 `begin = 0`，行为不变。
- 长 prompt 的 `begin > 0` 会启用已有的 query probe/replay 路径（chat 请求一直在走这条路），预填到 `N-1024` 后 replay 最后 1024 tokens。
- 该改动只影响 `/v1/completions` 或显式 token 化输入；NIAH 脚本走 `/v1/chat/completions`，不受影响。

## 3. 测试

### 3.1 单元测试

`kvmem_store_test` 通过。第二轮断言按 EMA 更新为 `0.45` / `0.5`。

### 3.2 64K 回归（59,336 prompt）

| 指标 | 基线（80% 水位） | 本次 | 回退后复测 |
|---|---:|---:|---:|
| prefill tok/s | ~400 | 402.2 | 435.3 |
| decode tok/s | 46.41 | 45.76 | 49.2 |
| draft 接受率 | 51.3% | 51.3% | 51.3% |
| 重选次数 | — | 8 | 8 |

decode 在接受率完全一致的前提下波动 ±4%，属于同参数噪声；8 次重选每次 576 页，transfer 0.17–0.98 s。

### 3.3 256K 召回（NIAH 33 针）

| 指标 | 基线（80% 水位） | 本次（EMA + query 尾部） |
|---|---:|---:|
| same-turn recall | 33/33 | **33/33** |
| final recall（strict） | 33/33 | **33/33** |
| final recall（union） | 33/33 | **33/33** |
| cross-tool（send_alert） | 14/14 | **14/14** |
| 工具调用总数 | 80 | **80** |
| 端到端（到 final sweep） | 637.1 s | **633.5 s / 588.1 s** |
| write prefill | ~390 tok/s | 343.6–928.8 / 396.8–999.8 tok/s |
| probe decode | 49.5 tok/s | 58.8–78.5 / 64.4–83.4 tok/s |

两列本次数据分别是 action-region 尝试回退前后的复测。召回与基线逐针一致；端到端差异来自 write prefill 的吞吐波动（同一份代码两次运行差 7%），属于运行间噪声，不是改动带来的差异。

### 3.4 `/v1/completions` 冒烟

- 2,666 prompt：200，2.72 s，decode 288.6 tok/s（短序列 + MTP 全接受）。
- 40,001 prompt：200，41.8 s，prefill 895.2 tok/s，未触发重选（低于窗口 53,248）。

两条 token-only 路径均正常，说明 `memory_query` 改为尾部窗口没有破坏 query probe/replay。

## 4. 回退

```bash
git checkout -- kvmem/include/kvmem/lrfu.hpp \
    kvmem/tests/kvmem_store_test.cpp \
    backends/ninfer/src/models/qwen3_5/frontend/frontend.cpp
```

或使用备份目录
`I:\llama\_backup\kvmem-llama.cpp-multi-backend-framework-20261020-143834`。

## 5. 后续

- 若 EMA 在 256K 上召回不降，可尝试 `--kvmem-lrfu-lambda 0.8`，把平均窗口从 2 轮拉到 5 轮。
- 若召回下降，先看 `KVMEM_RESELECT` 的选择集合是否抖动，再决定是否回退。
- 未做：真实 attention 打分（需要 kernel 暴露 attention）、per-layer 选择、in-place compaction。
