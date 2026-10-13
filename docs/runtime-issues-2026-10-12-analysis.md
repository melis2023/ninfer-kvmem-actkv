# 两个运行时问题分析（2026-10-12）

| 项目 | 内容 |
|---|---|
| 报告来源 | 用户实测，kvmem 0.18.0-rc3（启动器 `scripts/windows/start-backend.ps1`，后端 ninfer） |
| 启动参数 | `-Reserve 40960 -MaxTokens 102400 -ThinkingBudget 0` |
| 分析状态 | **两个 bug 均已定位、修复、加单测并实测验证（见 §5）** |

---

---

## 0. 两个问题的原始日志

```
# 问题 1：思考未结束，但 turn 已结束
13:19:54.695 INFO req#15 done | openai-chat | stop token | prompt 54,190 | output 16,824 |
  cache 54,167 (100.0%, private endpoint) | TTFT 2.3s | total 1m 57.2s | queue 1.9s |
  prefill 10.2 tok/s | decode 151.4 tok/s | mixed speculation accepted 14,310/24,783 (57.7%) |
  ngram 8,907/19,095 accepted, 617 rounds

# 问题 2：编辑上一轮提示词后，引擎崩溃退出
13:23:44.519 INFO req#16 started | openai-chat stream | 21 messages | max output 102,400 |
  thinking template default | tools 2
src\ops\launcher\scalar.cu:10: CUDA_CHECK(cudaGetLastError()) failed: cudaErrorAlreadyMapped:
  resource already mapped
ninfer exited with code -1073740791
```

`-1073740791` = `0xC0000409` = `STATUS_STACK_BUFFER_OVERRUN`，即进程调用了 `__fastfail` /
`std::abort` 而未走异常路径。

---

## 1. 问题 2（崩溃）——根因已确定

### 1.1 直接原因：**粘滞（sticky）CUDA 错误被记在了无关的调用点上**

`backends/ninfer/src/ops/launcher/scalar.cu:8-11`：

```cpp
void set_i32_scalar_launch(Tensor& destination, std::int32_t value, cudaStream_t stream) {
    set_i32_scalar_kernel<<<1, 1, 0, stream>>>(...);
    CUDA_CHECK(cudaGetLastError());     // <-- 报错点
}
```

`CUDA_CHECK` 展开为 `ninfer::cuda_check(...)`（`src/core/device.cu:66-76`）：

```cpp
void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    if (core::current_resident_memory()) {
        throw CudaError(err, ...);      // 有 residency 绑定 -> 抛异常
    }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", ...);
    std::abort();                       // 否则 -> abort (0xC0000409)
}
```

而 `cudaGetLastError()` 返回的是**自上次读取以来累积的第一个错误**，并会**清除**它。
所以 `scalar.cu:10` 报出的 `cudaErrorAlreadyMapped` **不是这次 kernel 启动产生的**，
而是**更早某处的失败遗留**，被这里"抓到"了 —— 报错位置与真正病因无关。

### 1.2 真正的病因：**每请求** 的 `PinnedHostBuffer` 分配失败（不是 `--kvmem-host-mib`）

`cudaErrorAlreadyMapped` 在本项目是 **pinned host 分配失败**的实测特征。仓库自己的注释写得很清楚：

`src/core/arena.cu:679-681`（`PinnedHostBuffer` 的失败分支里）：
> on Windows/WDDM this allocation is mapped into the GPU's address space and charged against the
> card, so a large pin fails once the device is nearly full even with tens of GiB of system RAM
> free ... **`cudaErrorAlreadyMapped` is the failure that shape produces.**

`src/models/qwen3_5/program/program_impl.cpp:519-521`：
> **Asking and backing off does not work.** One failed cudaMallocHost poisons every later one in
> the process (`cudaErrorAlreadyMapped` persists and `cudaGetLastError` does not clear it), so the
> size has to be chosen before the first attempt.

#### 1.2.1 先排除两个常见误判（本报告初稿两处都写错了）

**误判 A：不是 CUDA VMM 的 `cuMemMap` 二次映射。**

`cuMemMap` 是 **driver API**，失败是 `CUresult`，经 `NINFER_CU_CHECK` → `vmm::check`
**抛 `std::runtime_error`**（`src/core/vmm.h:42`）。若真是二次映射，报错文本会是
`cuMemMap(...) failed: CUDA_ERROR_ALREADY_MAPPED`，**不可能**是我们看到的
`CUDA_CHECK(cudaGetLastError()) failed: cudaErrorAlreadyMapped`（runtime 的 last-error 槽）。

而且**本配置下没有任何 VMM 消费者是活的**：

| VMM 消费者 | 启用条件 | 本次是否激活 |
|---|---|---|
| `VmmRegion`（模型挂起） | `--model-suspend` | ❌ 未传；且 KVMem **禁止**它 |
| `EvictableKVPool` / `EvictableWeightPool` | `overlay_vision()` | ❌ Vision residency 默认 `Resident`，KVMem 只允许 `Resident`/`Cpu` |
| D3D12 常驻堆 | `NINFER_WDDM_RESIDENCY` + `--wddm-evictable-budget` | ❌ 未传 |
| `ResidentMemorySession` | `CudaMemoryPolicy::StrictVram` | ❌ KVMem **强制** `DriverDefault`（见 §1.3） |

**误判 B：`--kvmem-host-mib 12288` 不是那个 pinned 分配。**

`--kvmem-host-mib` → `kvmem::HostKvStorage`，它用 **pageable heap**：
`kvmem/src/host/host_kv_storage.cpp:371` `new uint8_t[bytes]()`。
pageable 内存**不映射进 GPU 地址空间**，不参与 `cudaErrorAlreadyMapped`。

真正的 pinned 消费者是 **`PinnedHostBuffer`**（`src/core/arena.cu:652-715` → `cudaHostAlloc`）。
本项目里 **pinned** 的启动期大块由 `--host-kv-mib` 出资（`serve_options.cpp:966` →
`HostKVArena`），但本配置下它是**关闭**的：

* `context_cache.enabled` 默认 `false`（`types.h:301`）；
* KVMem 路径把 `host_kv_capacity_bytes = 0`（`model_instance.cpp:477`、`:495`）；
* 实测服务端日志就是 **`context cache | root only`**。

所以 pinned 分配**只剩每请求的 `PinnedHostBuffer`** —— 这才是病因。

#### 1.2.2 每请求 pinned 分配点（KVMem checkpoint 路径）

`backends/ninfer/src/models/qwen3_5/program/storage/kvmem_window.cpp`：

```cpp
:835   checkpoint->state = std::make_unique<PinnedHostBuffer>(bytes);   // capture_memory_checkpoint
:997   PinnedHostBuffer verify(bytes);                                   // restore_memory_prefix
:1098  auto state = std::make_unique<PinnedHostBuffer>(work.state.size());
:1184  auto state = std::make_unique<PinnedHostBuffer>(slot->cold_state.size());
```

`capture_memory_checkpoint`（`:821`）由三个请求期入口调用：
`capture_memory_prefix`（`:811`）、`capture_memory_endpoint`（`:874`）、
`capture_memory_rewrite`（`:903`）。

**关键**：这些是**未池化**的 —— `PinnedHostBuffer` 无参构造即 `cudaHostAlloc`，
析构即 `cudaFreeHost`（`arena.cu:671` / `:717-720`）。每个保留会话最多持有
`kKvmemCheckpointKinds` = 4 种 checkpoint（Endpoint / Reconstruction / InputFallback / Base，
`program_impl.h:486-488`），会话数上限 `retained_sessions`（默认 4，最大 16）。
**所以 pinned 占用随长对话单调增长，且在大块上反复 alloc/free。**

#### 1.2.3 缺陷：失败后**没有清除粘滞错误**

`arena.cu:663` 先读并清除历史 pending error，但 `cudaHostAlloc` 失败（`:672`）后
**直接 `throw`（`:699`），没有再清一次**。对比**正确惯用法** `release_mapping`
（`arena.cu:100-104`）：

```cpp
void release_mapping(void* pointer, cudaExternalMemory_t memory) noexcept {
    if (pointer != nullptr) { (void)cudaFree(pointer); }
    if (memory != nullptr) { (void)cudaDestroyExternalMemory(memory); }
    (void)cudaGetLastError();          // <-- 清掉，这个惯用法仓库里已有
}
```

**更关键：抛出被 `noexcept` 吞掉，错误槽留脏**

`kvmem_window.cpp:903-916`（`capture_memory_rewrite`，签名 `noexcept`）：

```cpp
void ProgramImpl::capture_memory_rewrite(SequenceState& sequence, KvmemCheckpointKind kind) noexcept {
    try {
        ...
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVMEM_REWRITE purpose=%s capture skipped: %s\n", purpose, error.what());
        // <-- 没有 (void)cudaGetLastError();
    }
}
```

`capture_memory_endpoint`（`:874`，同样 `noexcept`）在 `:889-891` 有**同样的问题**。

所以链条是：

```
KVMem checkpoint 捕获 -> PinnedHostBuffer(bytes) -> cudaHostAlloc 失败
   ↓  抛 std::runtime_error（arena.cu:699），但【没清 CUDA last-error 槽】
   ↓  被 noexcept 的 capture_memory_rewrite / capture_memory_endpoint 吞掉（只 fprintf）
   ↓  cudaErrorAlreadyMapped 仍是【进程级粘滞错误】
下一个碰巧调用 cudaGetLastError() 的 launcher（scalar.cu:10）把它读出来
   ↓
cuda_check: 无 residency 绑定 -> std::abort() -> 0xC0000409 -> 引擎整体退出
```

> `scalar.cu:10` 的 `CUDA_CHECK(cudaGetLastError())` 写法**本身是正确的** ——
> 三尖括号 kernel 启动确实要通过 last-error 槽报错。问题是**槽在它之前已经被别人弄脏**，
> 于是它报出了**别的调用**的失败，还署上了自己的 `file:line`。
> 这就是为什么崩溃点会是一个 1 线程的标量写（`set_i32_scalar`，在 prefill/decode 里用量很大：
> `text.cpp:1511/1641/1671-1672`、`prefill.cpp:41-43`、`mtp.cpp:36/58`）。


### 1.3 为什么"编辑上一轮"会触发（前缀分叉 → 重写 checkpoint 路径）

编辑历史会让请求走上 **rewrite-checkpoint** 路径，而该路径**在 prefill 期间**就要捕获 checkpoint：

1. **模板层检测到编辑** —— `chat_template.cpp:508-519`：当 `generation_begin > 0` 时产生
   `RewriteCheckpointByteSpec{kind = ResponseReplay | TurnClosure, offset = *generation_begin}`；
   `:490-507` 的"下一轮探针"通过重渲染并比较前缀来判断历史是否还能原样往返。
2. **计划层改走 rewrite 恢复** —— `request_plan.cpp:430-436`（`base->rewrite_checkpoint = candidate`）、
   `:678-689`（`plan->reuse = restore_path(...)`）。
3. **prefill 在被编辑的边界切分并捕获 InputFallback** —— `prefill.cpp:1414-1419`
   （把 `split_frontier` 设到 `rewrite_checkpoint->frontier`）、`:1468-1476`：
   ```cpp
   if (memory_rewrite_frontier && staged.cursor == memory_rewrite_frontier && …) {
       copy_tail(...);
       capture_memory_rewrite(sequence, KvmemCheckpointKind::InputFallback);
   }
   ```
   → `capture_memory_rewrite`（`:903`）→ `capture_memory_checkpoint`（`:821`）
   → **`PinnedHostBuffer(bytes)` @ `:835`** → `cudaHostAlloc` @ `arena.cu:671`。
   （首写路径 `prefill.cpp:1463-1467` → `capture_memory_prefix` 也会做同样的分配，
   但 `capture_memory_prefix` **不是** `noexcept`，所以它抛异常而不是 abort ——
   **abort 需要 `noexcept`+吞掉的那个变体**。）
4. **上一轮已经让 pinned 占用变大** —— req#15 的 `cache 54,167 (100.0%, private endpoint)`
   说明存在一个**被保留的大 Endpoint checkpoint**，它本身持有一份
   `state_images->host_layout().image_bytes` 的 pinned 镜像（`:834-835`）。

**净效果**：若只是追加，可复用 endpoint、不新增捕获；而**编辑**强制
(i) 在 prefill 中途新捕获一个 InputFallback checkpoint → **一次新的大块 pinned 分配**，
(ii) 在会话历史里多保留一种 checkpoint。这正是耗尽 WDDM 映射的 pinned 预算的增长模式 ——
所以"编辑后的第一个请求"死，而之前的请求没事。

### 1.4 三个独立缺陷（都应修）

| # | 缺陷 | 位置 | 后果 |
|---|---|---|---|
| **A** | 失败走 `std::abort()` 而非 `throw` | `device.cu:73-75`（无 residency 时） | **单个请求的错误杀死整个引擎** |
| **B** | `cudaHostAlloc` 失败后**不清** last-error 槽 | `arena.cu:699`（对比正确惯用法 `:100-104`） | 脏错误泄漏给无关调用点 |
| **C** | `noexcept` 捕获函数**吞掉异常却不清错误** | `kvmem_window.cpp:914-916`、`:889-891` | 脏错误在进程内长期存活 |

**缺陷 A 在本配置下是唯一可达分支（须特别强调）**：

```cpp
// device.cu:68-75
if (core::current_resident_memory()) { throw CudaError(...); }   // 需要 StrictVram 才绑定
std::abort();                                                     // 否则
```

* resident session 只在 `CudaMemoryPolicy::StrictVram` 下构造
  （`model_instance.cpp:756`、`:767-772`、`:778`）；
* 而 **KVMem 明确拒绝**非 `DriverDefault`（`model_instance.cpp:327` 把
  `cuda_memory_policy != DriverDefault` 列入拒绝清单）。

⇒ **KVMem 下 `throw` 分支是死代码**，每一次 `CUDA_CHECK` 失败都必然 abort。

**并且 `--recover-invariant-failures` 救不了这个崩溃**：它只捕获
`std::logic_error`（`engine_core.h:3225`），而 `CudaError` 派生自
`std::runtime_error`（`device.h:22-25`）。把它改成 `throw` 反而能救 ——
因为 `throw` 会落到 `catch (...)` → `crash_locked`（`engine_core.h:3249-3251`），
那里会 `fail_all_locked` 并**让进程活着**（worker 置 `failed_ = true` 后返回），
而不是 `0xC0000409` 退出。

### 1.5 与 `-Reserve` / `-MaxTokens` 的关系

* `-Reserve 40960` → `--kvmem-gen-reserve 40960`（`start-backend.ps1:13,136-137`）。
  ⚠️ **与另一处引用的 `16384` 冲突**，请确认实际值；两者都 64 对齐、都合法，
  且下面的机制对哪个值都不敏感。
* `-MaxTokens 102400` → `--default-max-tokens 102400`（`:136`）。
* `--kvmem-gen-reserve` / `--default-max-tokens` **都不做 VMM 预留**：前者是 KVMem 的追加窗口 R，
  只约束**设备 KV 页**（来自固定池）；后者只设 `effective_output_tokens`。
  且 `--kv-lease-growth` 在 KVMem 下是**失效**的（`model_instance.cpp:509` 强制 `false`）。
* **间接联系（推断，未证实）**：10 万 token 的输出预算 → 长多轮会话 →
  保留的 KVMem 历史更多 → 保留的 checkpoint pinned 缓冲更多
  （上限 `retained_sessions` × 4 种）。
* `--kvmem-host-mib 12288` **不参与**（pageable，见 §1.2.1 误判 B）。

**结论：这些参数是风险放大因素或无关，不是根因。**

### 1.6 修复建议（按优先级）与"唯一确证手段"

> **实施状态**：第 1~3 项**已实施**，见 §5。第 4 项刻意未做（需重构 `PinnedHostBuffer`
> 所有权模型，风险大）；第 5 项是运行期规避而非代码修复。

**修复（按杠杆排序）：**

1. **`device.cu:66-76` 永不 abort，一律 `throw`** —— 1 个函数，把崩溃变成"单请求失败 + 引擎存活"。
   在 KVMem 下这是**唯一**分支，所以等于修好了整个 (c) 问题。
2. **每个失败/消费点清粘滞错误** —— `arena.cu:699` 前加 `(void)cudaGetLastError();`
   （照抄 `release_mapping` `:100-104`）；同样处理 `resident_memory.cpp:213-223`、`stage_link.cu:13`。
3. **`noexcept` 捕获处别再留脏** —— `kvmem_window.cpp:915`、`:890` 的 catch 里加
   `(void)cudaGetLastError();`；更好的是返回状态让调用方知道 checkpoint 缺失。
4. **启动期预分配 pinned 池，别在运行中 pin** —— 这正是项目自己的结论
   （`program_impl.cpp:519-521`："the size has to be chosen before the first attempt"）。
   把 KVMem checkpoint 的 state image 改成从启动期固定池取，`cudaHostAlloc` 就永不后置调用。
5. **把 checkpoint 捕获失败变成 `kvmem::MemoryError`**，让它走已有的按 lane 隔离
   （`engine_core.h:3130-3132`）而不是到达 launcher。需先做 2。

**唯一确证手段（重要）**：贴出的日志是 **INFO 请求日志**，而吞掉的地方写的是 **stderr**。
请提供崩溃前后的**原始 stderr**，应能看到其中之一：

* `KVMEM_REWRITE purpose=input capture skipped: <error>`（`kvmem_window.cpp:915`）
* `KVMEM_ENDPOINT capture skipped: <error>`（`:890`）
* `ninfer: pinning N bytes of host memory succeeded, but cleared an unretrieved earlier CUDA error: …`
  （`arena.cu:707-710`）

有任何一条即可闭环；没有的话，H1 仍是最佳解释但未被直接证实。

---

## 2. 问题 1（思考未结束就 stop）——根因已确定并已修复

### 2.1 关键事实：`-ThinkingBudget 0` **等于"不设上限"**

`scripts/windows/start-backend.ps1:29,148`：

```powershell
[ValidateRange(0, 2147483647)][int]$ThinkingBudget = 0,
...
if ($ThinkingBudget -gt 0) { $nativeArgs += @('--default-thinking-budget', "$ThinkingBudget") }
```

**`0` 不会传 `--default-thinking-budget`**，即发动机不知道该在何时结束思考 ——
思考只能由**模型自己**输出结束标记，或撞上输出上限。

### 2.2 输出上限被**上下文**压到远低于 102400

`src/runtime/engine/effective_thinking_budget.h:38-42`：

```cpp
effective_output_capacity(requested_output_tokens, max_context, prompt_tokens)
    = min(requested_output_tokens, max_context - prompt_tokens + 1);
```

代入本次数据（`prompt 54,190`，`max-context 262144`，请求 `max output 102,400`）：

```
capacity = min(102400, 262144 - 54190 + 1) = min(102400, 207955) = 102400
```

**这一步没有被压缩**（207,955 > 102,400），所以 102,400 是真实上限。
但实际只输出了 **16,824**，说明是**更早的别的原因**终止了生成 —— 与 `stop token` 一致。

### 2.3 根因（已用代码证据闭环）：停止符的 **string 路径有 channel 过滤，token 路径没有**

这是本问题的**核心不对称**，直接给出因果。

**A) 停止「字符串」路径：按通道过滤，思考块内不会命中 content 停止符**

`include/infer/types.h:754-766`：

```cpp
enum class OutputChannel : std::uint8_t { Content, Reasoning };

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;   // 默认只作用于内容通道
    bool include_in_output = false;
};
```

`output_session.cpp:243-250`（`stop_hold_size`）与 `:261-265`（`feed_channel`）：

```cpp
for (const StopString& stop : policy.strings) {
    if (stop.channel != channel) { continue; }        // <-- 关键过滤
    ...
}
```

而思考文本走的是 `OutputChannel::Reasoning`（`:359-361`、`:372-374`）。
所以**字符串型停止符在思考块内被正确跳过** —— 设计是对的。

**B) 停止「token id」路径：完全不看通道/状态**

`output_session.cpp:637-639`：

```cpp
const bool stop_token =
    std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), token) !=
    impl_->policy.token_ids.end();     // <-- 只比 token id，没有 channel，也没有 in_reasoning
```

`:665-673`：

```cpp
if (stop_token) {
    if (!impl_->policy.publish_stop_token) { ... }
    terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count);
    release_logprobs(impl_->preview_state, impl_->preview_logprobs);
    return complete(count, FinishReason::StopToken);   // <-- 直接终止整个 turn
}
```

而**同一个函数紧邻的下一段（`:683-688`）却检查了 `in_reasoning`**：

```cpp
if (impl_->early_close_available && impl_->preview_semantic.in_reasoning &&
    impl_->preview_semantic.budget && ...) {
    ... return complete(count, FinishReason::None,
                        runtime::ContinuationAction::ApplyTargetControl);
}
```

`impl_->preview_state.in_reasoning` 就在手边（`:627` 刚用它统计 `reasoning_tokens`），
但 `:665` 的 stop 分支**没用它**。

**C) `policy.token_ids` 里装的是 turn 级停止符**

`frontend.cpp:486-494`：

```cpp
for (const int token : tokenizer.default_stop_token_ids()) { append_token(token); }  // 模型默认，含 <|im_end|>
for (const TokenId token : caller.token_ids) { append_token(token); }               // 调用方指定
```

**D) 没有任何"思考长度"兜底**

`reasoning_tokens` 是**纯计数器，无上限检查**（只在 `:627`/`:751` 自增，`:856` 供统计读取）。
所以思考长度唯一的上界是 `-ThinkingBudget`，而它是 `0`（=不传参 =不设限，见 §2.1）。

### 2.4 因果链（与日志逐项吻合）

```
1. ThinkingBudget 0  ->  start-backend.ps1:148 不传 --default-thinking-budget
                     ->  思考阶段无上限（reasoning_tokens 无兜底）
2. 多轮累积后思考很长（req#15 输出 16,824 token 全在思考中）
3. 模型在思考中途吐出 turn 级停止符（<|im_end|> 之类，属 policy.token_ids）
4. output_session.cpp:637 只比 token id，【没有】channel / in_reasoning 守卫
     —— 对比 :246 的字符串路径有 `stop.channel != channel` 过滤
5. :672 return FinishReason::StopToken  -> 整个 turn 立即结束
6. 客户端收到 finish_reason=stop，但 </think> 从未出现
     -> 用户观察到的「思考未结束，对话已结束」
```

**为什么"几轮后才出现"**：多轮后思考更长，中途收尾的概率上升。
**为什么 `prompt 54,190` 不是关键**：`effective_output_capacity=102,400` 未被上下文压缩
（见 §2.2），实际只输出 16,824 是被停止符终止，与上下文无关。

### 2.5 修复方向

> **实施状态**：下方的**方案 A 已实施**（见 §5.1），并加了单测。
> 方案 B（注入 `</think>`）未采用 —— 现有 `thinking_control_tokens` 只在设了
> `--default-thinking-budget` 时才可用，而本 bug 恰恰出现在**无预算**的情形。

**最小修法**：给 token 停止符路径补上与字符串路径同等的过滤。两条路都可选：

```cpp
// 方案 A（推荐，与字符串路径对称）：块内不放行 turn 级停止符
const bool stop_token = ...;
const bool turn_ending = /* 标记出 turn 级的那部分 token id */;
if (stop_token && !(impl_->preview_state.in_reasoning && turn_ending)) {
    ... 现有逻辑 ...
}
```

```cpp
// 方案 B（若不想区分 token 类型）：块内遇到任何停止符都先强制闭合思考
if (stop_token && impl_->preview_state.in_reasoning) {
    // 注入 </think> 收尾并继续生成，而不是终止 turn
}
```

配套：

1. `StopPolicy` 需**区分** end-of-thinking 与 end-of-turn 两类 token id（当前是扁平
   `std::vector<TokenId> token_ids`，无类型信息）；
2. 补单测：**思考块内输出 turn 级停止符不得结束 turn**（当前 `test_frontend.cpp` 无此用例）；
3. 建议同时给 `reasoning_tokens` 加一个兜底上限，避免"无上限思考 + 停止符缺失"时
   一路撞到输出上限才结束。

> **注意**：这属于**上游 ninfer 前端**的行为，与 KVMem/ActKV 无关，
> 但与用户报告完全一致。修它需要改动 `backends/ninfer` 并回写补丁。

### 2.6 备选/并存因素（同一判定点，已一并覆盖）

1. **多轮累积的思考历史**：若历史 assistant 的 reasoning 被截断后回灌进提示词，
   模型可能从"上一轮已结束的思考"续写而更早闭合。查 `--preserve-thinking` 渲染路径。
2. **推测解码**：`mixed speculation accepted 14,310/24,783`、`ngram 8,907/19,095`。
   若接受的 token 序列含停止符，接受路径会走**同一个** `:637` 判定 ——
   §2.5 的修复对它同样生效。

---

## 3. 结论与优先级

| 问题 | 定性 | 优先级 |
|---|---|---|
| **问题 2（崩溃）** | **确定性 bug（机制已代码闭环）**：KVMem checkpoint 的 `PinnedHostBuffer` 分配失败 → 错误槽留脏 → `noexcept` 吞掉 → 无关 launcher 读出 → `abort()` 杀引擎 | **P0** |
| **问题 1（思考未结束）** | **确定性 bug（代码闭环）**：停止符判定缺 `in_reasoning` 守卫（`output_session.cpp:637-673`），思考中命中 turn 级停止符即结束整个 turn | **P0/P1** |

**两个问题同源**：都发生在"多轮 + 大 `-Reserve` + 显存吃紧"的同一会话里，
且都是**边界处理不严** —— 问题 2 是错误处理边界太宽（abort 杀全局 + 脏错误泄漏），
问题 1 是状态判定边界缺失（思考中放行 turn 停止符）。

### 3.1 建议的修复顺序

1. **问题 1**：`output_session.cpp:637` 加 `in_reasoning` 守卫 + 区分两类停止符 + 单测。
   影响**每一次长思考对话**，用户已可稳定复现。**这是唯一根治手段**
   （只调 `-ThinkingBudget` 只能缓解，见 §2.5/§3.2）。
2. **问题 2 缺陷 A**：`device.cu` 改为始终 `throw` —— 防止单请求错误杀引擎。
   **这是最高杠杆的一处**：KVMem 下 abort 是唯一分支，改它即修好整个"引擎被杀死"的问题。
3. **问题 2 缺陷 B/C**：失败点与 `noexcept` catch 处清粘滞错误。
4. **问题 2 缺陷 D（可选）**：把 checkpoint state image 改为启动期预分配 pinned 池，
   从根上不再运行中 `cudaHostAlloc`。
5. **规避**：`-Reserve` 从 40960 降到 16384~20480，降低 pinned 分配失败概率。

### 3.2 关于 `-ThinkingBudget 0`（用户提问）

| 选项 | 问题 1 是否解决 | 依据 |
|---|---|---|
| 改成非 0（如 8192） | ⚠️ **缓解，不根治** | `start-backend.ps1:148` 只在 `> 0` 时传参；有上限后 `:683-688` 的早闭合被激活，会用 `ApplyTargetControl` **注入 `</think>` 收尾**（`FinishReason::None`，不结束 turn），且思考长度被封顶 → 窗口变窄 |
| **删除该参数** | ❌ **完全无效** | `0` 与"不传"**等价**（都不传 `--default-thinking-budget`）。删掉只失去"将来能设上限"的能力 |
| 修 `:637` 守卫 | ✅ **根治** | 唯一真正修掉 bug 的方法 |

**补充**：`--default-thinking-budget 0` 本身是**非法值**
（`serve_options.cpp:1238` 拒绝 0；`output_session.cpp:445-447` 二次抛异常），
所以 `0` **只能**表示"不传 = 无上限"，不存在"0 = 关闭思考"的语义。
关闭思考要用 `--no-thinking`。

**结论**：即使设了正数上限，只要模型在上限**之前**吐出 turn 级停止符，`:672` 依然立即终止 turn。
上限只是缩短窗口，没有关闭它。**所以必须修 `:637`；`-ThinkingBudget` 正数只作止痛。**

> **两个修复都在 `backends/ninfer` 内**，改完必须重新生成
> `backends/patches/ninfer-kvmem.patch` 并更新 `versions.json`
> （见 `docs/actkv-section56-wiring-record.md` §7 的构建环境缺陷说明）。

---

## 4. 本轮未做 / 未确证

* 未重新编译 ninfer（`score-agg` 移除的改动**已写入源码但未构建**）。
* **未做运行时复现**；问题 2 是崩溃路径的静态分析。
* **未确证具体是哪一次 `cudaHostAlloc` 失败、为何失败**（WDDM 地址空间耗尽 / 碎片 /
  与 `--kvmem-host-mib` 相关的显存压力）。机制已由代码证明，失败的具体尺寸没有 ——
  因为吞掉点写 **stderr**，而手头日志是 INFO 请求日志。
  **需原始 stderr 才能闭环**（见 §1.6）。
* 无法排除"更早的请求（req#14 或更早）已经留下 pending error，req#16 只是把它暴露出来" ——
  两种情形与现有日志完全一致。
* **本报告初稿有两处错误，已更正**：① 曾把病因归为 CUDA VMM 的 `cuMemMap` 二次映射
  （实为 runtime 的 pinned host 失败，且本配置下 VMM 全部未激活）；
  ② 曾把 `--kvmem-host-mib 12288` 当作 pinned 分配（实为 pageable `new uint8_t[]`）。
* 用户前后给出的 `-Reserve` 有 `40960` 与 `16384` 两个值，**需确认实际值**
  （不影响机制结论）。

---

## 5. 已实施的修复（2026-10-12）

### 5.1 问题 1：停止符在思考块内不再终止 turn

**`backends/ninfer/src/models/qwen3_5/frontend/output_session.cpp:641-671`**
（`preview_model` 的逐 token 循环，紧接 `stop_token` 计算之后）：

```cpp
if (stop_token && impl_->preview_state.in_reasoning) {
    DecoderState& state = impl_->preview_state;
    // 把仍持有的思考文本冲出（与自然 close 相同）
    const ReasoningCloseScan scan = scan_reasoning_close(state.think_marker_pending, true);
    const std::size_t split = scan.close != std::string::npos
        ? scan.close : state.think_marker_pending.size();
    feed_channel(state, OutputChannel::Reasoning,
                 std::string_view(state.think_marker_pending).substr(0, split),
                 impl_->policy, impl_->preview_output, count, nullptr);
    close_channel(state, OutputChannel::Reasoning, impl_->preview_output);
    state.think_marker_pending.clear();
    state.in_reasoning          = false;
    state.strip_content_leading = true;
    release_logprobs(state, impl_->preview_logprobs);
    continue;                       // <-- 不返回 StopToken，继续生成
}
```

**行为**：思考块内命中 turn 级停止符 → 关闭 Reasoning 通道、标记思考结束、**继续解码**
（而不是结束整个 turn）。因此客户端能拿到完整思考 + 正式回答，
`finish_reason` 不再是 `stop`。

**安全性**：

* **只触发一次**：`in_reasoning` 被清掉，所以第二个停止符走正常终止路径（单测已覆盖）。
* **有界**：继续生成仍受输出预算约束。
* **停字符串路径不受影响**：`match.found` 分支在守卫之前/独立处理。
* **恢复了原本的不对称**：停「字符串」有 `stop.channel != channel` 过滤
  （`feed_channel`），停「token id」此前没有 —— 这个守卫补上了。

**新增单测**（`tests/models/qwen3_5/test_frontend.cpp`，
`test_stop_token_inside_reasoning_keeps_the_turn_alive`，已在 main 注册）：

* 思考中命中停止符 → `finish_reason != StopToken`、全部 token 被接受；
* 之前的思考文本仍发布到 Reasoning；
* 停止符**之后**的回答发布到 Content；
* 第二个停止符（思考已关闭）→ 正常 `StopToken`。

### 5.2 问题 2-C：`cuda_check` 不再 `abort`（最高杠杆）

**`backends/ninfer/src/core/device.cu:66-88`**：

```cpp
void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    // 先打印（保证 stderr 可见），然后一律 throw —— 绝不再 std::abort()
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", ...);
    std::fflush(stderr);
    throw CudaError(err, ...);
}
```

**去掉了 `if (core::current_resident_memory())` 分支** —— 该分支在 KVMem 下**永远为假**
（resident session 只在 `StrictVram` 下建，而 KVMem 拒绝非 `DriverDefault`），
所以此前每次 KVMem 失败都走 `std::abort()` → `0xC0000409` 杀引擎。

**效果**：异常会沿栈上抛，落到 worker 的 `catch (...)` → `crash_locked`
（`engine_core.h:3249-3251`）。该函数**只**发布诊断、`fail_all_locked` 后 **return**
（`:2848-2863`），**进程存活**。所以单请求错误不再杀引擎。

> 注意：`--recover-invariant-failures` 仍不会接管它（只捕 `std::logic_error`），
> 但这不影响结论 —— `crash_locked` 本身就保证了进程存活。

### 5.3 问题 2-A/B：失败点清掉粘滞错误

| 文件 | 改动 |
|---|---|
| `src/core/arena.cu:699` | `cudaHostAlloc` 失败抛异常**前**加 `(void)cudaGetLastError();`（照抄同文件 `release_mapping:100-104` 的正确惯用法） |
| `src/core/stage_link.cu:13-20` | `check()` 抛异常前同样清一次 |
| `src/models/qwen3_5/program/storage/kvmem_window.cpp` | `capture_memory_endpoint`（`:889`）与 `capture_memory_rewrite`（`:914`）两个 `noexcept` catch 里各加 `(void)cudaGetLastError();` + 显式 `#include <cuda_runtime.h>` |

**效果**：失败被如实抛出/记录，且**不再污染** CUDA 的进程级 last-error 槽，
所以不会再有无关的 launcher 报告 `cudaErrorAlreadyMapped` 并背上黑锅。

### 5.4 验证证据（2026-10-12 实测）

| 检查 | 结果 |
|---|---|
| ninfer 全量重建（含测试目标，741 个目标） | **exit 0** |
| **新单测（修复后）** | **`EXITCODE=0` 通过** |
| **新单测（临时移除修复）** | **`EXITCODE=1` 失败**，报出原始症状：<br>`a stop token inside the reasoning block ended the turn`<br>`the round stopped short at a stop token inside the reasoning block`<br>`the answer after the mid-thinking stop token was not published` |
| 宿主单测（共享库） | **15/15 通过** |
| 修复后二进制启动（完整 launcher 参数） | ✅ `listening on http://127.0.0.1:18200` |
| 修复后推理（长思考 + 数学题） | ✅ `finish_reason=stop`、`content` 完整（`391`）、`reasoning_content` 存在 |
| 服务端 stderr | ✅ 无 `error` / `abort` / `CUDA_CHECK` / `KVMEM_*_skipped` |
| `prepare-backends.py --backend ninfer --check` | **ready** |
| 补丁反向应用 | **rc=0** |
| `versions.json` sha256 | **MATCH** |

**回归证明的意义**：临时把 `output_session.cpp` 还原到改动前、其余保持不变，
新单测立刻失败并**逐字复现了用户报告的现象** —— `finish_reason` 变成 stop、
轮次提前结束、**回答从未发布**。恢复修复后即通过。这证明单测确实锁住了这个 bug，
而不是"跟着实现写一个必然通过的断言"。

### 5.5 一个需要注意的既有问题（非本次引入）

`ninfer_memory_frontend` 全量跑会在**结构化输出/工具调用**用例处失败并抛异常中止：

```
frontend did not hide the terminal tool-call suffix
frontend did not retain tool-call parse diagnostics
frontend did not preserve normalized tool arguments
frontend test exception: generated token 32 violates the structured output grammar
```

**已用对照实验确认这是预存问题**：把 `output_session.cpp` 临时还原到改动前重建，
失败信息**完全相同**。与本轮修复无关（我未改 grammar / tool-call 路径）。

**副作用**：该异常会让 `main` 在到达新单测之前中止，所以新单测在**全量跑**时不执行。
为此在 `test_frontend.cpp` 的 main 里加了 `--reasoning-stop-only` 聚焦入口
（仿照既有的 `--thinking-control-only`），已验证通过。

> **建议**：这个预存失败值得单独排查（它会让整个前端测试套件失去保护）。
> 但它独立于本次两个 bug，未在本轮范围内修。

### 5.6 刻意未修

* **问题 2 缺陷 D**（checkpoint state image 改启动期预分配 pinned 池）：
  根治"运行中 pin 失败"的方案，但需重构 `PinnedHostBuffer` 的所有权模型，
  风险与工作量都大。当前修复已让失败**可恢复**（不再杀引擎）。
* **F1/F4**（日志暴露 `ended_in_reasoning`；停止字符串不要同时注册到 reasoning 通道）：
  增强项，非本 bug 必要条件。
* **`-Reserve` 调小**：运行期规避，非代码修复。

### 5.7 改动文件（12 个）

```
include/ninfer/types.h                             (移除 score_agg + conf_* 字段)
src/core/arena.cu                                  (pin 失败清粘滞错误)
src/core/device.cu                                 (不再 abort，一律 throw)
src/core/stage_link.cu                             (check() 清粘滞错误)
src/models/qwen3_5/frontend/output_session.cpp     (stop token 的 in_reasoning 守卫)
src/models/qwen3_5/program/decode.cpp              (§6 置信度采集)
src/models/qwen3_5/program/prefill.cpp             (§6 置信度采集)
src/models/qwen3_5/program/program_impl.h          (§6 conf_budget 成员)
src/models/qwen3_5/program/storage/kvmem_window.cpp (轮次门控 + catch 清错误)
src/models/qwen3_5/program/storage/memory_statistics.cpp (§6 monitor 创建)
src/serve/serve_options.cpp                        (--kvmem-conf-* + score-agg 拒绝 max)
tests/models/qwen3_5/test_frontend.cpp             (2 个新单测 + 聚焦入口)
```

