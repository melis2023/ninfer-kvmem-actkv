# ActKV 论文未实现功能盘点 + "5% watermark" 构想可行性分析

> 日期：2026-10-12
> 论文：《ActKV: Efficient LLM Agents through Action-Guided KV Cache Management》(arXiv:2609.31395v1)
> 本地：`I:\llama\kvmem-llama.cpp-multi-backend-framework`（ninfer 后端）
> 前置文档：[actkv-section56-wiring-record.md](actkv-section56-wiring-record.md)、[dev-actkv-swap-redesign.md](dev-actkv-swap-redesign.md)、[actkv-kvmem-port-analysis.md](../_ternary/README.md)

---

# 第一部分：论文功能 vs 本地实现（全量盘点）

## 0. 总账

论文三大组件 + 若干子机制，本地状态：

| # | 论文机制 | 论文位置 | 本地状态 | 说明 |
|---|---|---|---|---|
| 1 | 动作区注意力 → LRFU | §5.1 | ⚠️ **部分** | LRFU 有，但**打分源不是动作区** |
| 2 | AAV 访问模式分类 | §5.1 | ❌ **未实现** | cold-start/intermittent/continuous 三分未做 |
| 3 | top-p 90% hit 判定 | §5.1 | ✅ 已实现 | `lrfu_top_frac=0.90` |
| 4 | λ 衰减 LRFU 公式 | §5.1 | ✅ 已实现 | `λ=0.5`，A9 两阶段提交 |
| 5 | 动作区边界识别（模板匹配） | §5.2 Alg.1 | ❌ **未实现** | 用 query_sum（预填/用户 span）代替 |
| 6 | Token 置信度 T_i（top-k=20） | §6.1 | ✅ 已实现 | 复用引擎 `logprob_topk` |
| 7 | 滑窗 min-pool + bottom-b% | §6.1 | ✅ 已实现 | w=64, s=8, b=10% |
| 8 | 线性拟合斜率判趋势 | §6.1 | ✅ 已实现 | 一阶最小二乘 |
| 9 | 置信度驱动预算增长（×1.75） | §6.2 | ✅ 已接线 | 默认关闭；有冷却+幂等 |
| 10 | 预算**永不回缩** | §6.2 | ✅ 已实现 | 单测覆盖 |
| 11 | 预算增长**换来显存节省** | §6.2/§8.3 | ❌ **拿不到** | 池按 B+R 固定分配（见 §1.3） |
| 12 | 系统提示词 + 用户任务永不压缩 | §8.1 | ⚠️ **部分** | 只有 fixed sink/recent，非语义识别 |
| 13 | 恢复式注意力计算（LSE 复原） | §7.1 | ❌ **未实现** | 本地用 host 侧 mean-K 余弦 |
| 14 | 三原语标准化（计算/驱逐/压实） | §3.3 | ⚠️ **部分** | 驱逐有；另两个是等价物 |
| 15 | 原地压实 kernel（无冲突槽规划） | §7.2 | ⚠️ **有意不做** | KVMem 回收单位是块 slot，收益≈0 |
| 16 | GQA 组内 max（Alg.3 step 8） | §7.1 | ❌ **实测回退** | 39/40 → 28/40，改回 mean |
| 17 | 释放块归还调度器换并发 | §8.5 | ❌ **架构不同** | KVMem 单用户，无调度器竞争 |

**结论：核心可移植的都已做（§5 的 LRFU + §6 全套）；没做的是三类 —— ① 动作区语义识别；② 显存实际收益；③ §7 分页 kernel。**

---

## 1. 最有价值的缺口（按性价比排序）

### 1.1 ⭐ 动作区识别（§5.2 Alg.1）—— 这是论文的**根本前提**

论文原文（`actkv_full.txt:103`）：

> We identify the action region by **matching generated tokens to the LLM's standardized output template** to detect the start and end of action generation.

> We collect attention scores **within this region** and use them to update the temporal utility of KV entries after each ORA round.

**本地现状**：`attention.cpp:137-139` 的 `query_sums` 只在 `block != 1`（预填）时累积，来源是**预填 chunk 的 query**；`query_policy_user` 取的是 last-user span。也就是说：

| | 论文 | 本地 |
|---|---|---|
| 打分用的 query | **动作区**（tool-call / action 段） | **用户提问 span**（观测） |
| 语义 | "这个 KV 对**将来产生动作**有用吗" | "这个 KV 和**当前问题**相关吗" |

**这不是小差异，这是论文立论的基础。** 论文 §3.1 全部证据都在证明"action-preferred entries 和 observation/reasoning-preferred 是不同的集合"。用观测 span 打分，等于把 ActKV 退化成普通的 query-relevance 检索。

**已有设计**（`dev-actkv-swap-redesign.md:106-110`）明确列出三种口径，建议先用口径 2：

> 1. 上一轮 assistant 输出中 tool-call / action 段 token（需要解析）
> 2. 上一轮 assistant 输出的**尾部 N token**（默认 N = `block_tokens`，可配）
> 3. 边生成边累积的 `q_sum_`（需处理 pin 后窗口语义）

**建议**：先做口径 2（尾部 N token），成本极低，且能立刻 A/B。这是**唯一可能显著提升召回**的方向。

### 1.2 §7.1 恢复式注意力计算（LSE）—— 精度上限

论文的关键洞察（`actkv_full.txt:128-131`）：

> paged attention backends ... **do not expose the full attention matrix** needed for eviction... our key idea is to **recover attention scores from the softmax log-sum-exp (LSE)** produced by paged attention... we only recompute local query-key dot products within each paged KV block and normalize them with LSE.

即：`attn(i,j) = exp(s_ij − LSE_i)`，**LSE 是 flash attention 的免费副产品**，不需要物化整页 KV。

**本地现状**：`mean_key_retrieval_scores()` 用的是 **host 侧 mean-K vs mean-Q 余弦**（`kvmem/src/host/mean_key_retrieval.cpp`），不是真实注意力概率。

| | 论文 | 本地 |
|---|---|---|
| 分数 | `exp(q·k/√d − LSE)` 真实概率 | `cos(mean_q, mean_k)` 近似 |
| 成本 | LSE 免费 + 块内 q·k | host 侧 112 行 + 832 次互斥锁 |
| 语义 | **注意力质量**（和为 1） | 方向相似度（不归一） |

**可行性检查**：

- ✅ 引擎**已经有 LSE**：`src/ops/softmax_attention/dense/causal_cache/` 下 `causal_softmax_attention.cpp`、`prompt_bf16.cuh`、`prompt_fp8.cuh` 等多个文件命中 `lse`
- ✅ 可以做：把 LSE 暴露出来，块内重算 q·k
- ⚠️ **但风险高**：`dev-actkv-swap-redesign.md:104` 已判定"需要给 ggml 的 `ggml_flash_attn_ext` 加可选 LSE 输出"，属**阶段 B 单独立项**
- ⚠️ 而且 `mean` vs `max` 的 A/B 已经证明：**更"真"的信号不一定更好**（max 理论上更贴论文，实测 28/40）

**建议**：**不急**。先把 1.1 做完（成本低、收益可能大），LSE 留作 §1.1 做完后的精度上限实验。

### 1.3 §6 的显存收益 —— **结构性拿不到**

这是 `actkv-section56-wiring-record.md:63-72` 已如实记录的：

> **该功能换不到显存。** `pick_topk_blocks` 的窗口会变小，但 `MemoryBudget.selected_tokens` 仍是固定的 `--kvmem-budget`，物理池也按 B+R 分配。收益是"降低平均注意力窗口长度"，不是"降低显存占用"。

**根因**（已验证）：`model_instance.cpp:329` 强制
```cpp
options.kv_capacity.explicit_tokens == window * options.max_concurrency   // window = B + R
```
物理池在启动时按 `(B+R) × lanes` **静态分配**，`startup.cpp:1027` 用 `device_tokens()` 算 `logical_pages`。

**所以**：论文 §8.3 的"25.98% peak KV memory"本地**复现不了**。论文靠 vLLM 的**动态块池 + 归还调度器**；本地是**静态分区**。

要拿到收益需改池分配器（`model_instance.cpp`），属独立改动。**在单用户打满场景下收益≈0**（第 4 轮就填满池）。

### 1.4 系统提示词 / 用户任务保护（§8.1）—— 低成本补丁

论文（`actkv_full.txt:152`）：

> We **always preserve** the KV entries corresponding to the **system prompt and user task description**, excluding them from compression.

**本地现状**：只有 `--kvmem-sink-tokens`（默认 0 = 1 页 = 64 token）+ `--kvmem-recent-tokens`（默认 64）。这是**位置**保护，不是**语义**保护。

**问题**：系统提示词 + 工具 schema 在 dsh 场景常达 **数千 token**，64 token 的 sink 覆盖不了。一旦被驱逐，agent 会丢失任务约束。

**建议**：把 `sink_tokens` 显式设为"系统提示词 + 用户首轮"的实际长度（启动时已知）。**这是单行改动，且直接对应论文明确要求。**

### 1.5 §5.1 AAV 访问模式分类 —— 诊断工具

论文用 AAV 把条目分三类（cold-start / intermittent / continuous），并给出关键实测（`actkv_full.txt:96`）：

> Continuous entries are the most common, but their **per-access attention contribution is lower** than that of intermittent entries, which contribute strong attention in a discontinuous manner. Therefore, **evicting entries solely based on current action preference may mistakenly remove entries that will become important again.**

**本地现状**：无 AAV 实现，也没有"KS 集合跨轮 Jaccard 稳定性"指标。

**价值**：这**不是性能功能，是诊断功能**。它能回答一个本地一直没答的问题——**你的 λ=0.5 到底该多大**。论文的 λ 消融正是用 AAV 分类论证的。

**建议**：作为日志指标补上（`KVMEM_AAV_TRACE`），不参与决策。成本低，能指导调参。

### 1.6 §7.2 原地压实 —— 已论证**不应做**

`dev-actkv-swap-redesign.md:154` 的前提修正是对的：

> 论文 §7.2 的收益来自 **vLLM 式共享块池**——空洞使物理块无法归还调度器...KVMem 的回收单位是 block slot（`stage_out`+`free_slot` 已归还整块），**块内 token 级空洞不占池内存**且被刻意保留+掩码。

**所以 §7.2 本地收益≈0。** 除非"保留 cell / 占用 cell"空洞率 > 20%，否则不开工。**这个判断我复核后同意，不要做。**

### 1.7 GQA max（§7.1 Alg.3 step 8）—— 已实测回退，**不要重试**

`actkv-section56-wiring-record.md:129-130` 的 A/B 是硬的（同二进制、同机器、唯一变量）：

| 配置 | 同轮召回 | 最终 |
|---|---|---|
| `mean`（保留） | **39/40** | 39/40 |
| `max`（论文原意） | 28/40 | 32/40 |

原因分析（同文档 §5.2）也合理：max 取组内单 head 峰值 → 方差大 → 喂进 `KS_t = λ·KS_{t-1} + hit·score` 被 λ 记忆逐轮放大 → 渐进劣化（3/3→2/3→1/3）。

**注意**：这是论文与本地**唯一确证的分歧**。但本地样本是 NIAH 召回，论文是 ALFWorld/WebShop 任务成功率。**可能都对**——不必强行对齐。

---

## 2. 缺口清单（可执行）

| 优先级 | 缺口 | 改动量 | 预期收益 | 风险 |
|---|---|---|---|---|
| **P0** | 动作区识别（§5.2）→ 尾部 N token 口径 | 中（1 处打分源 + 参数） | **可能显著提召回** | 中（需 A/B） |
| **P1** | 系统提示词/任务保护（§8.1） | **小**（设 `sink_tokens`，或按语义 span） | 防丢任务约束 | 低 |
| **P1** | §6 开 `KVMEM_CONF_TRACE` 可观测 | 小 | 能看到预算走了几步 | 低 |
| **P2** | AAV / KS 跨轮 Jaccard 诊断 | 小 | 指导 λ 调参 | 低 |
| **P2** | §6 召回率 A/B（从未做过） | 中（12 min/轮） | 判定 §6 是否该开 | 低 |
| **P3** | LSE 恢复式打分（§7.1） | **大**（改 FA + 子模块 + patch） | 精度上限 | 高 |
| **不做** | §7.2 原地压实 | — | ≈0 | — |
| **不做** | 池分配器动态化（§6 显存收益） | 大 | 单用户≈0 | 高 |
| **不重试** | GQA max | — | 已证伪 | — |

---

# 第二部分："用 reserve 做 5% watermark"可行性分析

## 3. 先说结论

**不可行 —— 而且是"方向性"的不可行，不是实现细节问题。**

三个理由，全部有代码证据：

1. **R 不是空闲缓冲，它已经被完全占用**（§4.1）
2. **本地是静态分区池，不存在"抢占"这个对象**（§4.2）
3. **要的效果用现成的 `--kvmem-conf-budget` 更直接**（§4.3）

---

## 4. 逐条论证

### 4.1 R 已经 100% 被占用，没有 5% 可以"留出来"

你的参数：`--kvmem-budget 36864 --kvmem-gen-reserve 16384`。所以 `device_tokens() = B + R = 53248`。

**关键**：这个 53248 **不是容量上限，而是每轮实际使用的窗口**。看 `kvmem_window.cpp:576`：

```cpp
if (force_selection || compact_position(sequence, end) > kvmem_window_tokens) {
```

`compact_position = logical − removed_tokens`（`program_impl.h:1929-1935`）。也就是说：

- 窗口从 B=36864 开始
- 随着生成，`compact_position` 增长到 **B+R=53248**
- **只有超过 53248 才触发 reselect**

**所以 R 就是那个"增长余量"，它每一轮都会被用满。** 你说的"5% watermark"如果指"从 R 里留 5% 不分配"，那么：

- 相当于把触发阈值改成 `B + 0.95R = 36864 + 15565 = 52429`
- 效果：**reselect 提前 819 token 触发**
- 代价：**reselect 频率 +5%**

而 reselect 正是你最贵的操作——probe 实测 `advance` 中位 **415 ms**、最大 **1003.7 ms**，`select` 中位 135.8 ms。**每 16384 token 一次、每次阻塞 ~0.5 s**。

**多花 5% 的 reselect 换来的"缓冲"是假缓冲**：它不新增任何容量，只是把同一条线画早一点。

### 4.2 静态分区池里没有"抢占"可减少

vLLM 的 watermark 解决的是什么问题？**多请求竞争一个共享块池**：

```
请求 A、B、C 共享 GPU 块池 → 池将满时，新请求挤掉旧请求（preempt / swap out）
→ watermark 留出余量，避免"刚分配就要抢占"的抖动
```

**本地的结构完全不同**。`model_instance.cpp:329`：

```cpp
options.kv_capacity.explicit_tokens != window * options.max_concurrency
```

即池在**启动时静态切成 lanes 份**，每 lane 独占 `B+R`。加上默认 `max_concurrency = 1`（`serve_options.h:43`），你的场景是：

- **1 条 lane**
- **独占 53248 token 的窗口**
- **没有第二个请求来抢**

`pressure.cpp:2654-2668` 的准入检查针对的是 **host 侧 `memory_retained_budget`**（H=12288 MiB），不是 device 池：

```cpp
std::uint64_t available = memory_retained_budget;
...
if (base.impl_->memory_payload_reservation > available) return false;
```

**所以你要防的"抢占"在 device 池上不存在。** 真正会竞争的是 host H 预算（多会话保留历史时），而那个**已经有准入控制了**。

### 4.3 你想要的"自适应余量"，现成参数更直接

如果你的真实意图是"**别等到池满了才动作，留点余量**"，那么正确工具是 **§6 置信度预算**（已接线）：

```powershell
--kvmem-conf-budget 8192     # 起步小预算
--kvmem-conf-scale-up 1.75   # 论文值
--kvmem-conf-cooldown 4
```

它的行为正好是你想要的：**预算按置信度趋势动态增长，只在压缩边界变、永不回缩、有冷却**。区别是：

| | 你的 watermark 构想 | `--kvmem-conf-budget` |
|---|---|---|
| 触发依据 | **固定阈值**（5%） | **模型置信度**（信号驱动） |
| 方向 | 只留余量（容量↓） | 按需扩容量 |
| 论文依据 | 无 | §6.2 Alg.2 |
| 现状 | 需改代码 | **已实现、已测、默认关** |

**5% 是拍脑袋的常数；置信度是有依据的自适应信号。** 论文 §3.2 整节就在论证为什么固定 budget 在 agent 场景失效。

---

## 5. 如果你的意图是另外两个，结论不同

我要诚实说明：你的构想有三种可能读法，我只否定了第一种。

### 读法 B：防 reselect 撞上生成（不是防抢占，是**错峰**）

**这个有道理！** res elect 阻塞 ~0.5 s。如果它正好落在长输出中，用户会感到卡顿。

**但 watermark 解决不了**——它只改变触发时刻，不改变"reselect 发生时必须停下"这个事实。

**真正的解法**（本地已有）：**异步 reselect**。`kvmem_window.cpp:673-690` 的 `memory_io` 路径 + `allow_pending` 参数就是干这个的：

```cpp
if (memory_io) {
    auto session = std::make_unique<kvmem::MemorySession>(backend);
    session->prepare(...);
    pending = std::move(session);
    ...
    return prepare_window(sequence, begin, end, false, allow_pending);   // 不阻塞
}
```

**但它有个硬门槛**（`program_impl.cpp:465`）：

```cpp
if (kvmem_options.disk_bytes) {     // ← 只有启用 SSD tier 才创建 memory_io
    ...
    memory_io = std::make_unique<kvmem::StorageIoQueue>(...);
```

而你的 bat 没传 disk（第 78 行注释写的是"cold disk snapshots 被拒绝"）。

**⚠️ 这里有个文档与代码不一致，值得注意**：ninfer 的 `serve_options.cpp:1948` 实际**只要求两个 flag 一起给**：

```cpp
if (options.kvmem.disk_path.empty() != (options.kvmem.disk_bytes == 0) || ...)
    throw std::invalid_argument("KVMem SSD requires both --kvmem-disk-path and --kvmem-disk-mib");
```

即 **`--kvmem-disk-path` + `--kvmem-disk-mib` 同时给是被接受的**，不是被拒绝。`multi-backend.md:122` 里"被拒绝"说的是**多后端启动器/llama 侧**的行为，不是 ninfer 直连。**所以 bat 那条注释对 ninfer 路径可能是陈旧的。**

**这解释了你 68 个 probe 样本为什么全部 `async=0`**：没传 disk → `memory_io` 为空 → decode 走 `decode.cpp:490` 的同步 `prepare_window`（`allow_pending=false`）→ reselect 必然阻塞。

> **但先别高兴**：即使打开，收益也不确定——见 §6.2。

> ⚠️ **但这条要小心，它和已有结论冲突，且单 lane 下未必成立。**

`final-verification-report.md:64` 已经发现"异步只在 `--kvmem-disk-bytes` 启用时生效"，但随后的 §"为什么这里不该启用 SSD" 只用**容量**理由否决了它（11.2 GB 归档 < 12 GB `H` < 32 GB 内存 ⇒ "SSD 只会更慢"）。**那个否决没有考虑"开 disk 是为了打开异步流水线，而不是为了要容量"**——这是该论证的一个缺口。

**但反过来，单 lane 下异步也未必有用**，理由在 `engine_core.h:3125-3129`：

```cpp
if constexpr (requires { instance_.program->prepare_decode_memory(membership.sequences[0]); }) {
    ...
    if (!instance_.program->prepare_decode_memory(membership.sequences[row])) continue;   // ← 跳过这一 row
}
```

异步路径的做法是**把这条 row 跳过**（本轮不生成）。单请求时没有别的 row 可以填补这个空档，所以**墙钟代价可能和同步阻塞一样**。异步真正兑现收益的场景是**多 lane**——一条 lane 在搬运时其他 lane 还能算。

**所以结论应该是**：这是一个**未测的假设**，不是"最有价值的发现"。值得花一次实验证伪，但不要预设它会赢。

### 读法 C：给 host H 预算留余量

host `memory_retained_budget` 才是多会话真正竞争的资源，且 `pressure.cpp` 的准入是**硬拒绝**（`return false`），过冲会直接拒请求。

**这里加水印是合理的**，但正确做法是给 `--kvmem-host-mib` 留余量（调小），而不是动 device 的 R。而且 `--kvmem-retained-sessions`（默认 4）本身就是这个旋钮。

---

## 6. 结论与建议

### 6.1 直接回答

> **利用现有 reserve 分区做 5% watermark 来减少抢占 —— 不可行。**

因为：
1. **R 已被用满**：它是每轮增长的余量，不是空闲缓冲；从中切 5% 只等于把 reselect 提前 5%，**多付 5% 的 ~0.5 s 阻塞**
2. **没有抢占可减**：本地是 `(B+R) × lanes` 静态分区（`model_instance.cpp:329`）+ 默认单 lane（`max_concurrency=1`），不存在 vLLM 那种多请求争池
3. **要的效果有更好的现成工具**：`--kvmem-conf-budget`（信号驱动、有论文依据）>> 固定 5% 常数

### 6.2 但这个构想指向了一个**真问题**

你的直觉——"reselect 的代价应该被削平"——是**对的**。只是病根不在容量分配，而在**同步阻塞**：

| 现象 | 证据 | 病根 |
|---|---|---|
| reselect 阻塞 ~0.5 s | probe: advance 中位 415 ms | **decode 路径同步等待**（`decode.cpp:490` 用默认 `allow_pending=false`） |
| 68/68 样本 `async=0` | `final-verification-report.md:49` | `memory_io` 为空（`program_impl.cpp:465` 要求 `disk_bytes != 0`） |
| 异步 row 被跳过 | `engine_core.h:3129` `continue` | **单 lane 下空档无人填补** |

### 6.3 建议的下一步（排序）

| # | 动作 | 成本 | 预期 |
|---|---|---|---|
| 1 | **动作区识别**（§5.2 尾部 N token 口径） | 中 | **可能显著提召回**（唯一高收益方向） |
| 2 | **系统提示词/任务保护**（§8.1） | **低** | 防丢任务约束（可能解释此前的 HTML 质量损失） |
| 3 | 开 `--kvmem-conf-budget` 做召回 A/B（从未做过） | 中 | 判定 §6 是否该开 |
| 4 | 补 `KVMEM_CONF_TRACE` / AAV 诊断 | 低 | 可观测性，指导 λ 调参 |
| 5 | ~~加 disk 换取异步 reselect~~ | 低（改一行） | **存疑**：单 lane 下 row 被跳过，墙钟可能不变；仅在多 lane 时可能有效。**先测再信** |

**第 1 条优先**：它是论文立论的核心，本地用错了打分源（用户 span 而非动作区），且改动集中、可 A/B。

**第 5 条不要当灵丹妙药**：已有结论（SSD 容量冗余）和单 lane 跳过 row 两点都指向"可能没收益"。如果要测，用 `--kvmem-disk-mib` 给一个小配额打开 `memory_io`，只观察 `async=1` 是否出现、`advance` 是否真的离开关键路径——**别指望它解决卡顿**。

---

## 附：证据索引

| 结论 | 位置 |
|---|---|
| R=B+R 是每轮用满的窗口 | `kvmem_window.cpp:576`、`program_impl.h:1929-1935` |
| 池按 (B+R)×lanes 静态分配 | `model_instance.cpp:329`、`serve_options.cpp:1932,1939` |
| 默认单 lane | `serve_options.h:43` |
| host 准入（真正的竞争点） | `pressure.cpp:2654-2668`、`program_impl.cpp:473` |
| async 仅在 disk 时启用 | `program_impl.cpp:465-479` |
| probe 字段含 async=0/1 | `kvmem_window.cpp:680-720` |
| §6 已接线、默认关 | `confidence_monitor.hpp`、`types.h` conf_* |
| §6 拿不到显存 | `actkv-section56-wiring-record.md:63-72` |
| GQA max 实测回退 | `actkv-section56-wiring-record.md:129-130` |
| 动作区未实现 + 三种口径 | `dev-actkv-swap-redesign.md:106-110` |
| §7.2 不应做 | `dev-actkv-swap-redesign.md:154` |
| LSE 在本地存在 | `ops/softmax_attention/dense/causal_cache/*` |
| 打分用 query_sum（非动作区） | `attention.cpp:129-139` |
