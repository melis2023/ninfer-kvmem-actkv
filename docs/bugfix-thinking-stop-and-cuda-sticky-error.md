# 运行时 Bug 修复补丁：思考中停止符截断 + CUDA 粘滞错误杀引擎

| 项目 | 内容 |
|---|---|
| 日期 | 2026-10-12 |
| 仓库 | `I:\llama\kvmem-llama.cpp-multi-backend-framework`（后端 ninfer） |
| 分支 | `feat/multi-backend-framework` |
| 影响面 | **上游 ninfer 引擎**（`backends/ninfer`），与 KVMem/ActKV 无关 |
| 状态 | **已修复、已编译、已单测**；问题 1 已端到端复现，**问题 2 的触发条件未复现（见 §1.6）** |
| 补丁文件 | `backends/patches/ninfer-kvmem.patch`（sha256 `9db0f5c8…`，tree `4af8ba3d…`） |

> 这两个 bug 与 KVMem/ActKV 功能**无关**，是上游 ninfer 前端的错误处理缺陷，
> 但在长上下文多轮对话下被触发。它们可以独立于本分支上游化。

> ### ⚠️ 阅读须知：问题 2 的**触发条件未复现**（2026-10-12 复测）
>
> 问题 1（思考中停止符）**已确证**：单测对照可逐字复现原症状（§5.1）。
>
> 问题 2（CUDA 粘滞错误杀引擎）**只确证了错误处理缺陷本身（§1.2 的 A/B/C），没有确证触发条件**。
> 在一台 RTX 5060 Ti 16 GiB / 驱动 616.92 / 单卡 上，按 §1.6 的方法系统尝试后
> **未能复现崩溃**；而且实测**否证**了 §1.3 因果链里"pinned 分配计入显存预算"这一环
> （本机 pinned host 分配根本不进专用显存）。**§1.3 应读作"待证假设"，不是结论。**
> 详见 **§1.6**。

---

## 1. 问题 2：`-Reserve 40960` 下编辑提示词 → 引擎崩溃退出

### 1.1 原始报错

```
13:23:44.519 INFO req#16 started | openai-chat stream | 21 messages | max output 102,400 | tools 2
src\ops\launcher\scalar.cu:10: CUDA_CHECK(cudaGetLastError()) failed: cudaErrorAlreadyMapped:
  resource already mapped
ninfer exited with code -1073740791
```

`-1073740791` = `0xC0000409` = `STATUS_STACK_BUFFER_OVERRUN`（`__fastfail`/`std::abort` 的标志，
**不是**异常路径）。

### 1.2 根因：三层缺陷叠加

#### 缺陷 C（致命，最高杠杆）：`cuda_check` 用 `std::abort()` 而非 `throw`

`src/core/device.cu`，修改前：

```cpp
void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    if (core::current_resident_memory()) {
        throw CudaError(err, ...);      // 仅在有 residency 绑定时抛异常
    }
    std::fprintf(stderr, ...);
    std::abort();                       // 否则直接 abort
}
```

**关键**：`throw` 分支在 KVMem 下是**死代码**。

* resident session 只在 `CudaMemoryPolicy::StrictVram` 下构造（`model_instance.cpp`）；
* 而 **KVMem 明确拒绝**非 `DriverDefault`（`model_instance.cpp:327` 把
  `cuda_memory_policy != DriverDefault` 列入拒绝清单）。

⇒ **KVMem 下 `abort()` 是唯一可达分支**，任何一次 `CUDA_CHECK` 失败都杀死整个引擎。

> 附带发现：`--recover-invariant-failures` **救不了**这个崩溃 —— 它只捕获
> `std::logic_error`（`engine_core.h:3225`），而 `CudaError` 派生自 `std::runtime_error`。

#### 缺陷 A：失败点不清 CUDA 粘滞错误槽

`src/core/arena.cu` 的 `PinnedHostBuffer`，`cudaHostAlloc` 失败后**直接 throw**：

```cpp
throw std::runtime_error(cuda_error_message(prefix.c_str(), err));   // 槽仍是脏的
```

对比**同文件里已有的正确惯用法** `release_mapping`（清了一次）：

```cpp
void release_mapping(void* pointer, cudaExternalMemory_t memory) noexcept {
    if (pointer != nullptr) { (void)cudaFree(pointer); }
    if (memory != nullptr) { (void)cudaDestroyExternalMemory(memory); }
    (void)cudaGetLastError();          // ← 正确做法
}
```

#### 缺陷 B：`noexcept` 捕获吞掉异常却仍留脏

`src/models/qwen3_5/program/storage/kvmem_window.cpp`：

```cpp
void ProgramImpl::capture_memory_rewrite(...) noexcept {      // 注意 noexcept
    try {
        ...
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVMEM_REWRITE purpose=%s capture skipped: %s\n", ...);
        // ← 没有 (void)cudaGetLastError();
    }
}
```

`capture_memory_endpoint` 同样。

#### 为什么 `scalar.cu:10` 是"无辜背锅者"

`src/ops/launcher/scalar.cu`：

```cpp
void set_i32_scalar_launch(Tensor& destination, std::int32_t value, cudaStream_t stream) {
    set_i32_scalar_kernel<<<1, 1, 0, stream>>>(...);
    CUDA_CHECK(cudaGetLastError());     // 这行写法【本身是正确的】
}
```

kernel 启动确实要靠 last-error 槽报错。**问题在于槽在它之前已被别人弄脏** ——
`cudaGetLastError()` 返回的是**自上次读取以来累积的第一个错误**并清除它，
所以它报出的是**别的调用**的失败，还署上自己的 `file:line`。

### 1.3 完整因果链（⚠️ **前两环是待证假设，§1.6 已在本机否证**）

> **第 1~2 环（WDDM 计入显存 → `cudaHostAlloc` 失败）在 5060 Ti / 驱动 616.92 上不成立**：
> §1.6.4 实测 pinned 分配完全不进专用显存。第 3 环（rewrite 路径）**已复现**，
> 第 4~8 环（错误处理缺陷 → abort）**代码可证**。阅读时请把下面第 1~2 环当作"未证实的入口"。

```
[待证] Windows/WDDM 把 pinned host 内存映射进 GPU 地址空间并计入显存预算
   ↓  -Reserve 40960 使设备 KV 驻留更大 → 显存更接近上限
[已复现] "编辑上一轮" → 前缀改变 → 无法复用 checkpoint → 触发 rewrite 路径
   ↓  prefill.cpp:1414-1419/1468-1476 → capture_memory_rewrite
   ↓  kvmem_window.cpp:835  PinnedHostBuffer(bytes)   ← 每请求的 pinned 分配
[待证] arena.cu:671  cudaHostAlloc 失败 → cudaErrorAlreadyMapped
   ↓  抛异常但【不清错误槽】(缺陷 A)
   ↓  被 noexcept 的 capture_memory_rewrite 吞掉, 仍未清 (缺陷 B)
   ↓  错误在进程内【粘滞】
下一个调用 cudaGetLastError() 的 launcher (scalar.cu:10) 把它读出来
   ↓  无 residency → std::abort() (缺陷 C) → 0xC0000409 → 引擎退出
```

### 1.4 修复

**修复 C**（`src/core/device.cu`）：

```diff
 void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
     if (err == cudaSuccess) { return; }
+    // Always throw; never std::abort(). A CUDA failure inside one request must
+    // fail that request, not kill the process.
+    //
+    // The resident-session branch used to be the only throwing one, and it is
+    // UNREACHABLE under KVMem: a ResidentMemorySession is built only for
+    // CudaMemoryPolicy::StrictVram (model_instance.cpp) and KVMem rejects every
+    // policy except DriverDefault. So every KVMem run took the abort branch, and
+    // an ordinary per-request allocation failure exited the whole engine with
+    // 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN).
+    //
+    // Throwing is never worse than aborting: a noexcept caller would have called
+    // std::terminate() either way, while a normal caller now unwinds to the
+    // worker's catch(...) handler, which fails the active requests and keeps
+    // serving. The diagnostic is still printed first so the error is visible on
+    // stderr even when the throw is caught somewhere up the stack.
     std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                  cudaGetErrorName(err), cudaGetErrorString(err));
-    std::abort();
+    std::fflush(stderr);
+    throw CudaError(err, std::string(file) + ":" + std::to_string(line) +
+                            ": CUDA_CHECK(" + expr + ") failed: " +
+                            cudaGetErrorName(err) + ": " + cudaGetErrorString(err));
 }
```

**效果**：异常沿栈上抛 → 落到 worker 的 `catch (...)` → `crash_locked`
（`engine_core.h:3249-3251`）。该函数**只**发布诊断、`fail_all_locked` 后 **return**
（`:2848-2863`），**进程存活**。**单请求错误不再杀引擎。**

**修复 A**（`src/core/arena.cu`，`PinnedHostBuffer` 失败路径）：

```diff
+        // The failed cudaHostAlloc just set the CUDA runtime last-error slot. Clear it before
+        // throwing, exactly as release_mapping does above: the slot is process-global and sticky,
+        // so leaving it dirty makes an unrelated later call report THIS failure under its own
+        // file:line. That is precisely how a failed pin here used to surface as
+        // "scalar.cu:10: CUDA_CHECK(cudaGetLastError()) failed: cudaErrorAlreadyMapped" and abort
+        // the engine. The failure is still reported through the throw.
+        (void)cudaGetLastError();
         throw std::runtime_error(cuda_error_message(prefix.c_str(), err));
```

**修复 B**（`kvmem_window.cpp` 两处 `noexcept` catch + `stage_link.cu`）：

```diff
--- kvmem_window.cpp: capture_memory_endpoint
     } catch (const std::exception& error) {
+        // A skipped checkpoint is survivable, but the CUDA runtime last-error slot is not: it is
+        // process-global and sticky, so a failure swallowed here resurfaces later under an
+        // unrelated call's file:line. Clear it, then keep going without the checkpoint.
+        (void)cudaGetLastError();
         std::fprintf(stderr, "KVMEM_ENDPOINT capture skipped: %s\n", error.what());
     }

--- kvmem_window.cpp: capture_memory_rewrite
     } catch (const std::exception& error) {
+        // See capture_memory_endpoint: a skipped checkpoint is survivable, a dirty CUDA
+        // last-error slot is not. Clear it before continuing.
+        (void)cudaGetLastError();
         std::fprintf(stderr, "KVMEM_REWRITE purpose=%s capture skipped: %s\n", purpose, error.what());
     }

--- stage_link.cu: check()
 void check(cudaError_t err, const char* what) {
     if (err == cudaSuccess) { return; }
+    // Clear the CUDA runtime last-error slot before throwing. It is process-global and sticky, so
+    // a failure reported through the throw must not also resurface later under an unrelated call's
+    // file:line. Mirrors release_mapping in arena.cu.
+    (void)cudaGetLastError();
     throw std::runtime_error(std::string("StageLink ") + what + ": " + cudaGetErrorName(err) +
                              ": " + cudaGetErrorString(err));
 }
```

另加 `#include <cuda_runtime.h>` 到 `kvmem_window.cpp`（显式化，此前靠传递包含）。

### 1.5 运行期建议（非代码）

`-Reserve` 从 `40960` 降到 `16384~20480`，降低 pinned 分配失败概率。
**修好缺陷 C 后，失败只会让该请求跳过 checkpoint，不再杀引擎。**

### 1.6 复现尝试（2026-10-12）：**本机未复现**，且否证了 §1.3 的关键一环

复现环境：**单卡** `GPU-dba26dd9-…`，RTX 5060 Ti，16,311 MiB，驱动 **616.92**。
保留了修复前的二进制（`exe-step3-1501.exe`，step3 构建，`--kvmem-score-agg` 仍在）
作为对照组，避免"用已修版本复现不出"的循环论证。

#### 1.6.1 先用 `--kvmem-verify-transfers` 量出 pinned 块大小

```
KVMEM_PREFIX_CHECKPOINT frontier=52  state_bytes=153954304      # 146.8 MiB
```

`PinnedHostBuffer(bytes)` 的 `bytes = state_images->host_layout().image_bytes`
（`kvmem_window.cpp:836-837`），本模型 **≈ 147 MiB**，每个 checkpoint 一块。
一次请求里可见 4 次捕获（input / reconstruction / prefix / endpoint），即瞬时最多 ~4×147 MiB。

#### 1.6.2 尝试过的全部组合（修复前二进制）

| 模型 | `--kv-dtype` | `-Reserve` | 结果 |
|---|---|---|---|
| Coder390-Q2LynnStyle（12.5 GiB 权重） | nvfp4 | **40960**（用户值） | ❌ **根本起不来**：`requires 2294315264 bytes, but only 1701363712 available` |
| Coder390-Q2LynnStyle | nvfp4 | 16384 | ✅ 起，**free 468 MiB** |
| Coder390-Q2LynnStyle | nvfp4 | 20480 | ✅ 起，**free 402 MiB** |
| Coder390-Q2LynnStyle | nvfp4 | **24576** | ✅ 起，**free 388 MiB**（该模型能起的最紧配置） |
| Coder390-Q2LynnStyle | nvfp4 | 28672 | ❌ 起不来 |
| GSQ-RCO-IQ3_S（11.3 GiB） | nvfp4 | **40960** | ✅ 起，**free 1.26~1.30 GiB** |
| GSQ-RCO-IQ3_S | nvfp4 | 81920 | ✅ 起，**free 637 MiB** |
| GSQ-RCO-IQ3_S | nvfp4 | 94208 | ✅ 起，**free 327 MiB** |
| GSQ-RCO-IQ3_S | int8 | 40960 / 81920 | ❌ 起不来（int8 要 3.56 / 5.04 GiB） |

**在每一个能起来的配置上都跑了同一套多轮剧本，全部无崩溃：**

* `-Reserve 40960`（IQ3_S）：10 轮 → **20 条消息 / 65,215 token 提示词**，再发 3 次
  **改历史**的请求；
* `-Reserve 81920` / `94208`（free 637 / 327 MiB）：同上；
* 最紧的 Q2LynnStyle + `-Reserve 24576`（free 388 MiB）：3 轮 + 2 次改历史。

**改历史的路径确实走到了**（这一点被证实）：追加请求 `TTFT 14s`、`cache 88~90%`；
而改动历史后 `TTFT 1m 40s`、`cache 0 (0.0%)` —— 前缀在编辑点分叉，**全量重 prefill**，
正是 §1.3 描述的 rewrite 路径。

**但 stderr 里没有任何失败**：

```
$ Select-String 'KVMEM|capture skipped|CUDA_CHECK|AlreadyMapped' repro-*.err
（无匹配）
```

进程全程存活，`KVMEM_*_capture skipped:`（**未受 `verify_transfers` 门控，失败必打印**）
一次都没出现。

#### 1.6.3 从外部制造显存压力：也无效，而且是**有意义的无效**

用驱动 API（`nvcuda.dll` 的 `cuMemAlloc`，worker 目录没有 `cudart64_*.dll`）在**另一个进程**
里占显存：

| 动作 | nvidia-smi 视角 |
|---|---|
| 占用 512 MiB | used 15,333 → 15,801，**free 718 → 250 MiB** |
| 之后发一次正常请求 | ✅ 正常返回，147 MiB 的 pin **照常成功** |
| 试图占 3000 MiB | used 只 +196 MiB → **大部分落进了 shared memory** |

结论：这台机器上**无法从外部制造"专用显存"压力**——WDDM 把超出专用预算的分配溢到系统内存。

#### 1.6.4 决定性实测：pinned host 分配**根本不进专用显存**（否证 §1.3）

在最紧配置（Q2LynnStyle + `-Reserve 24576`，引擎启动即 `free 389 MiB`）下，
请求前后连续采样 20 秒：

```
idle                : used 15,682 MiB | free 369 MiB
request (2,717 tok) : used 15,665 ~ 15,682 MiB   <-- 峰值与空闲一致
idle after          : used 15,665 MiB | free 386 MiB
```

该请求会创建 ~4 块 147 MiB（合计 ~588 MiB）的 pinned checkpoint，**专用显存用量一格没动**。
若 pinned 分配真如 upstream 在 24 GiB RTX 3090 上测得那样"计入显卡预算"，
588 MiB 的 pin 在只剩 370 MiB 的空间里必然失败或至少把 used 顶满。**它没有。**

⇒ **§1.3 因果链中"WDDM 把 pinned host 内存计入显存预算"这一环，在本机不成立。**
upstream 的 3090 表（`src/core/arena.cu:679-681`、`program_impl.cpp:519-521`）是另一套硬件的实测，
不能直接搬到 5060 Ti / 驱动 616.92。**因此 `cudaHostAlloc` 不是本机可证的触发点。**

#### 1.6.5 顺带查实的两个配置问题（对用户直接有用）

`I:\llama\ninfer-server-rtx50-sm120a-ui\ninfer-server-gui-config.json`（最后保存 2026-10-10 21:31）
**不是崩溃时（10-12 13:19）的有效配置**：

1. 它写的 `model_path` 是 `Coder390-Q2LynnStyle-MTP`，而该模型配 `-Reserve 40960` **起不来**
   （见 1.6.2 第一行）；
2. 它的 `extra_args` 里还留着 `--kvmem-score-agg max` —— 修复后是**启动错误**：
   ```
   ninfer-serve: --kvmem-score-agg max was removed: it regressed the 40-needle / 256k
   recall from 39/40 to 28/40 ... GQA aggregation is always the folded mean now; drop the flag.
   ```
3. 它同时设了 `fast_prefill_kernel: true` + `kv_dtype: nvfp4` —— 修复后也是**启动错误**：
   ```
   KVMem requires ... INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8 fast prefill ...
   ```
   （`--help` 里"nvfp4 + FP4 Tensor Core QK kernel"指的是非 KVMem 路径。）

**用 GUI 起服务前，先把这三处清掉**（`--kvmem-score-agg` 整个删掉；`Fast prefill` 取消勾选
或把 KV 类型改成 `int8`；确认模型与 reserve 的组合能起来）。

#### 1.6.6 这次尝试真正得到了什么

| 结论 | 状态 |
|---|---|
| 缺陷 C（KVMem 下 `cuda_check` 只能走 `abort()`）| ✅ 代码可证（死代码分支 + 拒绝清单） |
| 缺陷 A/B（失败不清 CUDA 粘滞槽、被 `noexcept` 吞掉）| ✅ 代码可证 |
| 修复方向正确（单请求错误不该杀进程）| ✅ 逻辑成立，且是纯防御性改动 |
| `cudaHostAlloc` 因 WDDM 显存压力失败 | ❌ **本机否证** |
| 崩溃可复现 | ❌ **未复现** |

**所以问题 2 的原始触发点仍未知。** 修复的价值在于：
**下一次它再发生时，报错会指向真正失败的调用**（arena.cu 的 throw 文本 + 不再留脏槽），
而且**只让该请求失败、引擎继续服务**，不会再是 `0xC0000409` 整个进程退出。

> **要在你的机器上确证，需要崩溃当时的原始 stderr。** 现在的修复版会打印
> `KVMEM_REWRITE purpose=… capture skipped: …` 或 `KVMEM_ENDPOINT capture skipped: …`
> 并带上失败的分配大小与 `device has … MiB free`；若看到的是别的 `CUDA_CHECK` 失败，
> 那才是真凶，`scalar.cu:10` 只是背锅者。

---

## 2. 问题 1：多轮后思考未结束，turn 却已结束

### 2.1 原始现象

```
13:19:54.695 INFO req#15 done | openai-chat | stop token | prompt 54,190 | output 16,824 |
  cache 54,167 (100.0%, private endpoint) | ... | mixed speculation accepted 14,310/24,783 (57.7%)
```

用户描述："提示词达到 54190，输出过程还在思考时输出 16824，思考未结束但对话已结束。"

### 2.2 根因：停止符判定缺 `in_reasoning` 守卫（两条路径不对称）

`src/models/qwen3_5/frontend/output_session.cpp:preview_model()`

**停止「字符串」路径有通道过滤（设计正确）**：

```cpp
// types.h
struct StopString {
    std::string text;
    OutputChannel channel = OutputChannel::Content;   // 默认只作用于 content 通道
};

// output_session.cpp:feed_channel
for (const StopString& stop : policy.strings) {
    if (stop.channel != channel) { continue; }         // ← 关键过滤
    ...
}
```

思考文本走 `OutputChannel::Reasoning`，所以**字符串停止符在思考块内被正确跳过**。

**停止「token id」路径完全没有过滤**（修改前）：

```cpp
const bool stop_token =
    std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), token) !=
    impl_->policy.token_ids.end();      // ← 只比 token id。没有 channel，也没有 in_reasoning
...
if (stop_token) {
    terminalize(...);
    return complete(count, FinishReason::StopToken);   // ← 直接终止整个 turn
}
```

**三处铁证**：

1. **同一函数紧邻的下一段（`:683`）检查了 `in_reasoning`** —— 说明作者知道这个状态存在：

   ```cpp
   if (impl_->early_close_available && impl_->preview_semantic.in_reasoning && ...)
   ```

2. **`in_reasoning` 就在手边** —— `:627` 刚用它统计 `reasoning_tokens`：

   ```cpp
   if (impl_->preview_state.in_reasoning) { ++impl_->preview_state.reasoning_tokens; }
   ```

3. **`policy.token_ids` 装的是 turn 级停止符**（`frontend.cpp:486-494`）：

   ```cpp
   for (const int token : tokenizer.default_stop_token_ids()) { append_token(token); }  // 含 <|im_end|>
   ```

**为什么"几轮后才出现"**：多轮后思考更长，模型中途收尾的概率上升。
**为什么 54190 不是关键**：`effective_output_capacity = min(102400, 262144-54190+1) = 102400`，
未被上下文压缩。实际只输出 16,824 是**被停止符终止**，与上下文无关。

### 2.3 修复

**核心改动**：块内命中停止符时，**不终止 turn**，而是关闭思考块并继续生成。

```diff
         const bool stop_token =
             std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), token) !=
             impl_->policy.token_ids.end();

+        // A stop token that arrives while the reasoning block is STILL OPEN is not a turn
+        // terminator. The token id list holds the model's turn-level end-of-turn ids, and a long
+        // reasoning phase can emit one before "</think>" ever appears. Honouring it here ended the
+        // turn mid-thought: the client got finish_reason=stop with no close marker and no answer,
+        // and reasoning_tokens accounted for the whole output. Treat it as the model's implicit end
+        // of thinking instead: close the reasoning channel, mark the reasoning phase done, and keep
+        // decoding so the turn still produces an answer.
+        //
+        // Note the asymmetry this restores. Stop STRINGS are filtered per channel
+        // (`stop.channel != channel` in feed_channel), so a content stop string never fires inside
+        // the reasoning channel. Stop TOKEN IDS had no such filter. This guard supplies it.
+        //
+        // The repair fires at most once per turn: in_reasoning is cleared below, so a second stop
+        // token goes through the ordinary terminal path. The output budget still bounds the
+        // continuation. The byte/stop-string paths (`match.found`) are unaffected.
+        if (stop_token && impl_->preview_state.in_reasoning) {
+            DecoderState& state = impl_->preview_state;
+            // Flush whatever reasoning text is still held, exactly as the natural close does.
+            const ReasoningCloseScan scan = scan_reasoning_close(state.think_marker_pending, true);
+            const std::size_t split =
+                scan.close != std::string::npos ? scan.close : state.think_marker_pending.size();
+            feed_channel(state, OutputChannel::Reasoning,
+                         std::string_view(state.think_marker_pending).substr(0, split),
+                         impl_->policy, impl_->preview_output, count, nullptr);
+            close_channel(state, OutputChannel::Reasoning, impl_->preview_output);
+            state.think_marker_pending.clear();
+            state.in_reasoning          = false;
+            state.strip_content_leading = true;
+            release_logprobs(state, impl_->preview_logprobs);
+            continue;                       // <-- 不返回 StopToken，继续生成
+        }
+
         DecoderState before_state;
```

**安全性**：

* **只触发一次**：`in_reasoning` 被清除，第二个停止符走正常终止路径（单测覆盖）。
* **有界**：继续生成仍受输出预算约束。
* **停「字符串」路径不受影响**：`match.found` 分支独立处理。
* **恢复了原本的不对称**：给 token id 路径补上与字符串路径同等的过滤。

---

## 3. 附带修复：移除 `--kvmem-score-agg max`

**理由**（实测 A/B，40 针 / 256k，同二进制同机器）：

| 配置 | 同轮召回 | 最终召回 | 工具调用 |
|---|---|---|---|
| **`mean`** | **39/40** | **39/40** | 80/81 |
| `max`（论文 Alg.3 step 8） | 28/40 | 32/40 | 68 |

逐轮同轮召回：

```
mean: 4/4 3/3 3/3 3/3 3/3 3/3 3/3 3/3 2/3 3/3 3/3 3/3 3/3
max : 4/4 3/3 3/3 3/3 3/3 2/3 2/3 1/3 1/3 2/3 2/3 1/3 1/3
                              ^ 第 6 轮起渐进劣化
```

`max` 取组内单个最强 head，分数方差更大；喂进 LRFU（`KS_t = λ·KS_{t-1} + hit·score`）
后噪声逐轮放大，action-critical 块逐渐丢槽位。

**改动**：删除 `score_agg` 字段与 `MeanKeyGroupReduce::Max` 分支，GQA 聚合固定为
**折叠均值**。传 `--kvmem-score-agg max` 现在会**明确报错**（而非静默接受）：

```
--kvmem-score-agg max was removed: it regressed the 40-needle / 256k recall from 39/40 to
28/40 (docs/actkv-section56-wiring-record.md). GQA aggregation is always the folded mean now;
drop the flag.
```

涉及：`kvmem/include/kvmem/mean_key_retrieval.hpp`、`kvmem/src/host/mean_key_retrieval.cpp`、
`backends/ninfer/{include/ninfer/types.h,src/serve/serve_options.cpp,src/models/qwen3_5/program/storage/kvmem_window.cpp}`、
`src/adapter/llama-memory-kvmem.{h,cpp}`、`src/adapter/llama-kvmem-hooks.h`、
`tools/llama-kvmem-{cli,server}.cpp`、`scripts/run-dialog-arm.ps1`、`workers/ninfer/start-kvmem-ninfer.bat`。

---

## 4. 如何应用本补丁

### 4.1 已有准备好的 ninfer 树（推荐）

补丁已固化在 `backends/patches/ninfer-kvmem.patch`，直接校验：

```powershell
cd I:\llama\kvmem-llama.cpp-multi-backend-framework
python scripts\prepare-backends.py --backend ninfer --check
# 期望: ninfer: 8319e8f51247 + ninfer-kvmem.patch ready
```

### 4.2 从零准备

```powershell
python scripts\prepare-backends.py --backend ninfer     # init + apply patch
```

### 4.3 编译（本机已验证）

```powershell
# 需要 VS 环境 + vcpkg 变量（否则 FFMPEG 找不到）
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
  -version '[17.0,18.0)' -latest -products * `
  -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
& "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$env:VCPKG_ROOT='I:\vcpkg'; $env:VCPKG_TARGET_TRIPLET='x64-windows'

# ⚠️ 改了头文件（types.h），必须全量 touch —— 本构建目录的 C++ 头依赖追踪是坏的
Get-ChildItem backends\ninfer -Recurse -Include *.cpp,*.h,*.cu,*.cuh -File |
  Where-Object { $_.FullName -notmatch '\\third_party\\' } |
  ForEach-Object { $_.LastWriteTime = Get-Date }

cmake -S backends/ninfer -B build-backends/ninfer -DBUILD_TESTING=ON
ninja -C build-backends/ninfer ninfer-serve ninfer_memory_frontend
```

### 4.4 跑测试

```powershell
# 专项测试（本修复）
build-backends\ninfer\tests\ninfer_memory_frontend.exe --reasoning-stop-only
# 期望 exit=0

# 宿主单测
ctest --test-dir build-kvmem-host
# 期望 100% tests passed, 15/15
```

### 4.5 回滚

```powershell
# 只回滚代码（保留补丁归档）
git -C backends/ninfer checkout -- src/core/device.cu src/core/arena.cu src/core/stage_link.cu `
  src/models/qwen3_5/frontend/output_session.cpp `
  src/models/qwen3_5/program/storage/kvmem_window.cpp tests/models/qwen3_5/test_frontend.cpp
```

> ⚠️ `backends/patches/ninfer-kvmem.patch` 包含**仓库原有的未提交工作**
> （例如 `InvalidToolConstraint` 枚举扩展、`--kvmem-conf-*` 等），
> 直接 `git checkout` 补丁会让它与工作树不一致，需重新生成并更新 `versions.json`。

---

## 5. 验证证据

| 检查 | 结果 |
|---|---|
| ninfer 全量重建（含测试，741 目标） | **exit 0** |
| **新单测（修复后）** | ✅ **exit 0 通过** |
| **新单测（临时移除修复）** | ❌ **exit 1 失败**，逐字复现原症状（见下） |
| 宿主单测 | **15/15 通过** |
| **40 针 / 256k 召回（修复后）** | ✅ **39/40 同轮、39/40 严格、39/40 重试并集**，81 工具调用 —— 与 `mean` 基线**逐位一致**，含同一根漏针（`N26`） |
| 232k 上下文 / 49 请求全程 stderr | ✅ **0** 条 `error`/`abort`/`CUDA_CHECK`/`skipped`/`AlreadyMapped`；引擎全程存活 |
| 修复后二进制启动（完整 launcher 参数） | ✅ `listening on http://127.0.0.1:18200` |
| 修复后推理 | ✅ `finish_reason=stop`、`content` 完整（`391`）、`reasoning_content` 存在 |
| 服务端 stderr | ✅ 无 `error`/`abort`/`CUDA_CHECK`/`KVMEM_*_skipped` |
| `prepare-backends.py --check` | **ready** |
| 补丁反向应用 | **rc=0** |
| `versions.json` sha256 | **MATCH** |
| **问题 2 触发条件复现** | ❌ **未复现**（9 种配置 + 外部显存压制 + 65k 多轮改历史，全部无崩溃；见 §1.6）|

### 5.1 回归证明（关键）

把 `output_session.cpp` 临时还原到改动前、其余不变，新单测**立刻失败**并复现原始症状：

```
a stop token inside the reasoning block ended the turn
the round stopped short at a stop token inside the reasoning block
the answer after the mid-thinking stop token was not published
```

恢复修复后即通过。**这证明单测锁住的是真 bug，而不是"跟着实现写的必然通过断言"。**

### 5.2 新增单测

`tests/models/qwen3_5/test_frontend.cpp`：

* `test_stop_token_inside_reasoning_keeps_the_turn_alive` —— 思考中命中停止符 → 不结束 turn、
  思考文本保留、之后的回答被发布；
* `test_stop_token_after_reasoning_still_ends_the_turn` —— 思考**已闭合**后的停止符仍正常终止
  （防止修复变成"全面禁止停止符"）；
* main 内新增 `--reasoning-stop-only` 聚焦入口（仿既有 `--thinking-control-only`）。

### 5.3 端到端回归：40 针 / 256k 召回（修复后实测）

用**修复后的 worker**（`workers\ninfer\infer-kvmem-server.exe`，2026-10-12 18:05:59）
重跑既有的 40 针 / 256k 工具，配置与 `mean` 基线完全相同
（`--kvmem-budget 36864 --kvmem-gen-reserve 16384`，KV 53,248 tokens，int8，`--kvmem-action-lrfu`，
`--kvmem-recent-tokens 1024`，`--model-id ninfer-27b`）：

```powershell
powershell -File I:\dsh\html\run-recall-256k.ps1     # 起服务（脚本内联完整参数）
python I:\dsh\html\niah40_256k.py                    # 40 针 / 256k，约 13 分钟
```

| 指标 | `mean` 基线（修复前） | **修复后** |
|---|---|---|
| 同轮召回 | 39/40 | **39/40** |
| 最终召回（严格首答） | 39/40 | **39/40** |
| 最终召回（1 次重试并集） | 39/40 | **39/40** |
| 工具调用（同轮 / 最终 / 重试） | 40 / 40 / 1 = 81 | **40 / 40 / 1 = 81** |
| 漏掉的针 | `N26`（`N26-063352`） | **同一根 `N26`** |

逐轮同轮召回（与基线**逐轮相同**）：

```
修复后: 4/4 3/3 3/3 3/3 3/3 3/3 3/3 3/3 2/3 3/3 3/3 3/3 3/3
基线  : 4/4 3/3 3/3 3/3 3/3 3/3 3/3 3/3 2/3 3/3 3/3 3/3 3/3
```

**服务端健康度**（232k 上下文、49 个请求、约 11 分钟连续压测）：

```
$ Get-Content recall256k-postfix-server.err |
    Select-String 'error|abort|CUDA_CHECK|skipped|AlreadyMapped'
（无匹配）
```

引擎全程存活（同一 pid、未重启），无任何 CUDA 粘滞错误泄漏，
`req#48/#49` 仍在 232,672 token 提示词上正常作答。

**为什么应当逐位一致**：该测试用 `reasoning_effort: none` 发请求，根本不进思考通道，
所以问题 1 的守卫在这条路径上不会被触发；问题 2 的三个修复只改动**错误处理路径**，
正常路径一条指令都没变。因此"逐位一致"正是期望结果，它证明：
**修复不改变召回行为，只把致命崩溃变成单请求失败。**

> 吞吐差异（本轮 prefill 363 tok/s、轮墙 47.5s，基线 384 tok/s、45.0s）来自机器负载，
> 与本补丁无关：改动都在错误路径上。python 退出码 `1` 是该工具的设计
> （`return 0 if n_after_retry == len(final_hits) else 1`），代表"有 1 根针未恢复"，
> **不是崩溃**；基线同样是 `1`。

> 说明：这条测试是**追加式**对话，不会触发"编辑提示词 → 前缀改变 → rewrite"路径，
> 因此它不能复现问题 2 的崩溃现场，只能证明修复后**无回归 + 引擎在 232k 下稳定**。
> 崩溃路径的验证见 §5.1（回归隔离）与 §1 的因果链。

---

## 6. 已知的既有问题（非本补丁引入，未修）

`ninfer_memory_frontend` **全量**跑会在结构化输出/工具调用用例处失败并抛异常中止：

```
frontend did not hide the terminal tool-call suffix
frontend did not retain tool-call parse diagnostics
frontend did not preserve normalized tool arguments
frontend test exception: generated token 32 violates the structured output grammar
```

**已用对照实验确认是预存问题**：把 `output_session.cpp` 还原重建，失败信息**完全相同**。

**副作用**：该异常让 `main` 在到达新单测前中止 —— 这就是新单测需要
`--reasoning-stop-only` 聚焦入口的原因。

**建议单独排查**：它会让整个前端测试套件失去保护，但独立于本补丁的两个 bug。

---

## 7. 未修项（刻意）

| 项 | 原因 |
|---|---|
| 问题 2 缺陷 D：checkpoint state image 改启动期预分配 pinned 池 | 根治"运行中 pin 失败"，但需重构 `PinnedHostBuffer` 所有权模型，风险/工作量都大。当前修复已让失败**可恢复** |
| 日志暴露 `ended_in_reasoning`（F1） | 增强项，非本 bug 必要条件 |
| 停止字符串不要同时注册到 reasoning 通道（F4） | 独立的第二条件截断路径，应单独评估 |
| `-Reserve` 调小 | 运行期规避，非代码修复；**且本机复测后这条建议缺乏依据**（§1.6） |
| 问题 2 的真实触发点 | 未定位。需要崩溃当时的原始 stderr（§1.6.6）；现有信息不足以继续收敛 |

---

## 8. 改动文件清单

**本补丁的两个 bug 修复（6 文件）**：

```
backends/ninfer/src/core/device.cu                                    # 缺陷 C: 不再 abort
backends/ninfer/src/core/arena.cu                                     # 缺陷 A: pin 失败清错误
backends/ninfer/src/core/stage_link.cu                                # 缺陷 B: check() 清错误
backends/ninfer/src/models/qwen3_5/frontend/output_session.cpp        # 问题 1: in_reasoning 守卫
backends/ninfer/src/models/qwen3_5/program/storage/kvmem_window.cpp   # 缺陷 B: 两处 catch 清错误
backends/ninfer/tests/models/qwen3_5/test_frontend.cpp                # 2 个新单测 + 聚焦入口
```

**同批归档（移除 score-agg，见 §3）**：另外 6 个文件。

---

## 9. 一句话总结

**问题 1** 是**状态判定缺陷**（已确证、已复现）：停止符的 token 路径缺少字符串路径已有的通道过滤，
思考块内命中 turn 级停止符就结束了整个 turn，**回答从未发布**。

**问题 2** 是**错误处理缺陷**（已确证、已修）：任何一次 `CUDA_CHECK` 失败，在 KVMem 下
只能走 `std::abort()`（`throw` 分支是死代码），加上失败时不清 CUDA 进程级粘滞错误槽、
又被 `noexcept` 吞掉，于是**一个请求的 CUDA 错误就杀死整个引擎**，还让 `scalar.cu:10` 背锅。

> **但它的触发条件没复现**：本机 9 种配置 + 外部显存压制都没能让 `cudaHostAlloc` 失败，
> 而且实测 pinned 分配**不进专用显存**，否证了"WDDM 计入显存预算"这一环（§1.6）。
> 所以 `-Reserve 40960` 是风险放大因素这一说法，目前**没有证据支持**。
> 修复的价值在于**把不可归因的进程退出，变成可归因、可恢复的单请求失败**。
