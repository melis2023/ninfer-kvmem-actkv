# 2026-10-25 上游整合评估与进度

## 1. 基准

| 项 | 值 |
|---|---|
| 本地工作分支 | `dflash2-kvmem-fixes-20261018` @ `927fa93a` |
| Neroued master | `64d10fbf`（2026-10-16，164 个提交本地没有） |
| Neroued dev | `a522cfef`（2026-10-24，336 个提交本地没有） |
| 共同祖先 | `f76e19c0`（2026-09-17） |
| 本地 fork 链 | `Neroued/ninfer` → `Don-Chad/ninfer-3090` → `iamwavecut/ninfer-all` → 本地 |

本地相对主线有 1117 个提交，ops 目录布局和 context 架构都不同，**不能整体 merge**。

## 2. 备份

- 子模块：分支 `backup/pre-integration-20261025` + tag `backup-pre-integration-20261025`（`927fa93a`）
- 主仓库：分支 `backup/pre-integration-20261025` + tag（`193b4ad`）
- 工作二进制：`I:\model\backup\20261025-pre-integration\ninfer-kvmem-server-927fa93a.exe`
- 启动 bat 与 `BUILD-INFO.json` 同目录

## 3. 已完成：Jinja trim 的 cache boundary

来源：`a8e212ac`（trim 优化）+ `68c54356`（cache boundary）。

本阶段只取 boundary 修复，不取 `trim_utf8` 性能改动，避免改 `src/text/unicode.*`。

改动 6 文件，+77/−3：

| 文件 | 内容 |
|---|---|
| `third_party/llama-jinja/jinja/string.h` | `string_part::collapsed_source_end`、`append_boundary`、`cut_bytes(..., bool)` |
| `third_party/llama-jinja/jinja/string.cpp` | `tag` 清理折叠段；`append`/`cut_bytes` 传播边界；`strip` 调 `cut_bytes(..., true)` |
| `third_party/llama-jinja/jinja/value.cpp` | `tojson` 只在唯一非折叠 part 上保留 origin |
| `src/text/jinja.h` | `TemplateBoundaryMapping`、`TemplateOutput::boundary_mappings` |
| `src/text/jinja.cpp` | render 时收集折叠边界 |
| `src/models/qwen3_5/frontend/prompt_layout.cpp` | `source_boundary()` 查询 boundary mappings |

验证：

- `ninfer-serve` 目标构建通过（CUDA 13.4 / sm_120a）；
- 服务器启动：`capacity 49,152 tokens, nvfp4, explicit | runtime 1.36 GiB | free 847.4 MiB`；
- 短请求正常：`17*23` → `391`，`draft 34/40`；
- build id：`backup-pre-integration-20261025-dirty`。

待补：用带 `trim_blocks`/`{%-` 的模板做 prefix cache 命中对比（当前服务器用的是产物内置模板）。

## 4. 已完成：Graph 启动校准（`19367582`）

上游：59 文件，+952/−373。核心：

1. `StartupMemoryBudget`（initial/limit/headroom/retained）；
2. `DeviceAllocationError` + `device_memory_snapshot/growth`；
3. `ProgramImpl::initialize_graphs()` / `release_graphs()` 拆分；
4. `make_graph_probe_plan`（最小 KV 池、host context 0、graph_probe）；
5. `calibrate_graphs` 量 `rebuild_bytes` / `retained` / `preparation`；
6. 用 `available - rebuild` 重规划 KV/expert cache；
7. 正式 Program 实际占用检查 + 一次纠偏重试。

本地阻塞点：

- 本地 `construct_model` 已长到 1026 行，含 strict VRAM、WDDM、multi-rank、vision overlay、graft pages、residency 等逻辑；
- 本地用 `CudaError` + `ResidentMemorySession` 做分配失败分类和 strict 重试，和上游的 `DeviceAllocationError`/`StartupMemoryBudget` 不是一套；
- 本地 KVMem 的 host context / device KV 页规划和上游 context 不同，探针 `host_capacity_bytes = 0` 的语义要重新定义；
- 本地已有 `--cuda-graph-allowance-mib` 手动覆盖，可先用来估实际 Graph 占用。

评估：核心思想可移植，但不是 59 文件直接贴；建议单独一个变更，先做 dense 路径（27B 无 MoE），MoE 的 `ResourcePurpose::GraphCalibration` 后置。

### 4.1 手动实测（不改代码）

用本地已有的 `--cuda-graph-allowance-mib`，q2blocks / q3blocks 开图：

| 模型 | B/R | 总 | allowance | 结果 |
|---|---:|---:|---:|---|
| q2blocks | 21,504 / 15,360 | 36,864 | 默认 | ✅ 2.09 GiB，free 1.27 GiB |
| q2blocks | 32,768 / 16,384 | 49,152 | 默认 | ✅ 2.30 GiB，free 1.04 GiB |
| q2blocks | 36,864 / 16,384 | 53,248 | 默认 | ✅ 2.37 GiB，free 992 MiB |
| q2blocks | 40,960 / 16,384 | 57,344 | 默认 | ❌ 差 ~7 MiB |
| q2blocks | 40,960 / 16,384 | 57,344 | 700 MiB | ✅ 2.18 GiB，free 964 MiB |
| q2blocks | 45,056 / 16,384 | 61,440 | 700 MiB | ✅ 2.25 GiB，free 835 MiB |
| q2blocks | 49,152 / 16,384 | 65,536 | 700 MiB | ✅ 2.32 GiB，free 903 MiB |
| q2blocks | 53,248 / 16,384 | 69,632 | 700 MiB | ✅ 2.39 GiB，free 732 MiB |
| q2blocks | 57,344 / 16,384 | 73,728 | 700 MiB | ❌ 差 ~58 MiB |
| q3blocks | 32,768 / 16,384 | 49,152 | 700 MiB | ✅ 2.04 GiB，free 991 MiB |
| q3blocks | 40,960 / 16,384 | 57,344 | 650 MiB | ✅ 2.13 GiB，free 623 MiB |

结论：

- 探针实测 Graph 仅 **115–143.5 MiB**；盲预留约 1 GiB；
- 规划下限取 **256 MiB**，等于拿回约 700–800 MiB；
- q3blocks + 开图 + **77,824** 可用，free 457 MiB，decode 8.7K **66.3–66.7 tok/s**，接受率 43.6%（draft 5）；
- q3blocks 开图极限约 **86,016**（free 313 MiB）；90,112 差 13 MiB；
- q2blocks + 开图 + 69,632 也可用，free 631 MiB。

### 4.3 已实现（dense 最小闭环）

改动 5 文件：

| 文件 | 内容 |
|---|---|
| `src/models/qwen3_5/program/program.h` | `measure_cuda_graph_bytes()` 声明，返回 `optional<size_t>` |
| `src/models/qwen3_5/program/program.cpp` | 构造最小探针 Program，读 `cuda_graph_measured_bytes`；失败返回 `nullopt` |
| `src/models/qwen3_5/program/planning/startup.h` | `make_graph_probe_plan()` 声明 |
| `src/models/qwen3_5/program/planning/startup.cpp` | 探针计划：DFlash2 用 KVMem 窗口作逻辑容量，物理页取 `page_count(window)`，绕过容量曲线最小值 |
| `src/runtime/engine/model_instance.cpp` | 最终容量规划前调用探针；实测值加 256 MiB 下限后写回 `cuda_graph_allowance_bytes`；失败回退盲预留 |

行为：

- `--cuda-graph-allowance-mib N` 仍优先，跳过探针；
- `--no-cuda-graph` 直接返回 `nullopt`，无额外启动开销；
- WDDM 把 free delta 夹到 0 时，探针成功但 measured=0，仍用 256 MiB 下限（不会退回盲预留）；
- 探针失败（异常）才回退盲预留。

实测日志：

```text
CUDA graph probe measured 115.0 MiB; planning allowance 256.0 MiB
CUDA graphs ready | 1.2s
capacity | KV 77,824 tokens, nvfp4, explicit | runtime 2.10 GiB | free 456.9 MiB
```

当前 bat：

```bat
set "MODEL=...q3blocks.ninfer"
--kvmem-budget 61440 --kvmem-gen-reserve 16384 --kvmem-host-mib 12288
--cuda-graph-allowance-mib 已移除（自动校准）
```

### 4.4 未做

- 上游的一次纠偏重试（Device/Graph OOM 后按 recovered 再规划一次）；
- MoE 的 `ResourcePurpose::GraphCalibration`（本地 `moe_experts`/`sparse_moe` 接口不同）。

### 4.2 建议的 dense 校准实现

本地已经有一部分基础：`ProgramImpl` 构造时用 `device.free_bytes()` + residency dedicated delta 测了 `graph_measured_bytes`，并在 `MemorySummary` / 日志中报告（`program_impl.cpp:710-731`、`types.h:1623`）。缺的是**把探针测量反馈到 `cuda_graph_allowance_bytes`**，而不是从零实现上游的 `GraphCalibration`。

不做上游 59 文件全量，先做最小闭环：

1. 用 `make_sequence_planner(...).finalize(1)` 或等价的 `build_sequence_candidate(inputs, 1)` 构造最小物理 KV 计划（`capacity` 仍是 max_context，Graph profile 覆盖完整）；
2. 构造探针 `ProgramImpl`，读 `memory_summary().cuda_graph_measured_bytes`；
3. 销毁探针，把 `options.cuda_graph_allowance_bytes` 设为实测值（加下限，如 256 MiB）；
4. 走现有 `construct_model` 的容量解析；
5. 探针失败或测量为 0 时回退盲预留；
6. 不做一次重试/容量纠偏（后续再加）。

这样不改 `ProgramImpl` 的构造/析构契约，也不动 strict/WDDM 的容量解析主路径。

本地 `moe_experts`/`sparse_moe` 的接口与上游 `src/ops/moe` 不同，MoE 的 `ResourcePurpose::GraphCalibration` 不能直接复制。

## 5. 待定：qwen4-exp + MoE 全套

规模：

| | 本地 | 上游 master |
|---|---:|---:|
| `src/models/qwen4_exp` | 19 文件 / 3,854 行 | 75 文件（execution/program/ple/state） |
| MoE ops | 14 文件 / 4,426 行（`moe_experts`/`moe_route`/`sparse_moe`） | 72 文件（40 kernels + 15 routed_swiglu + 8 topk_softmax_swiglu_moe + 4 route） |

上游 qwen4-exp 依赖：

- `src/ops/linear` 的 family 重构（projection/add/swiglu/kernels/split/pair/topk）；
- 新的 context/program 架构（`b9114396`）；
- `src/ops/moe` 的 routed expert / CPU offload / CUDA graph 体系。

本地已有自己的 Flash-Next/qwen4-exp 路径（`executor.cpp` + expert cache/stream + ngram），但和上游不是同一实现。

三种策略：

| 方案 | 内容 | 风险 | 周期 |
|---|---|---|---|
| A. 分支级合并 | 把上游 master 合到新分支，重放 KVMem/DFlash2/sm75 | 极高 | 长 |
| B. 选择性移植 | 保留本地 qwen4-exp，只补缺失的 MoE 内核/后端（CPU NVFP4、block-FP8、routed_swiglu 形状） | 中 | 中 |
| C. 保持现状 | 本地 Flash-Next 继续用现有实现，只修 bug | 低 | 短 |

### 5.1 合并冲突实测

在临时 worktree 上执行：

```text
git merge --no-commit --no-ff neroued/master
CONFLICT files: 891
```

891 个冲突文件，覆盖 `apps/`、`bench/`、`src/ops/`、`src/models/`、`tools/convert/`。结论：**A 方案不是补丁级工作，是重新整合分支**，不建议在当前阶段做。

### 5.2 B 方案的可移植清单

按依赖从低到高：

1. `src/ops/moe/kernels/cpu/nvfp4`（AVX2/AVX512 BF16）——本地 `moe_experts` 只有 bf16/gguf；
2. `src/ops/moe/kernels/cuda/block_scaled/{fp8,nvfp4}`——block-FP8 / NVFP4 expert；
3. `routed_swiglu` 的 q4_q5 / q4_q6 / q8_q8 / nvfp4 / fp8 形状；
4. `topk_softmax_route` / `topk_softmax_swiglu_moe` 的 h2048/h2560 形状；
5. 本地 `sparse_moe` 的 decode/prefill/small_t 路由接这些新 kernel；
6. qwen4-exp `execution/program` 的 eager text/vision/MTP + CUDA graph，最后再接。

每步都可以单独构建、单独跑 Flash-Next 回归，不需要动 KVMem。

注意：本地 `moe_experts` / `sparse_moe` 的接口与上游 `src/ops/moe` 的 `routed_swiglu` / `topk` 不同，不能直接复制文件；要按本地的 `PreparedExecution` / `ExpertBank` / `sparse_moe_*` 接口适配。

建议先 B，除非明确要上游 qwen4-exp 的全部行为。

## 6. 待定：Context / runtime 重写（`b9114396`）

上游：176 文件，+21k/−32k。替换 context cache，加 preemptive scheduling、private continuations、shared checkpoints、snapshot/replay recovery、per-request TTFT。

本地 KVMem 是另一套 context 管理（host/device 分页、LRU、window recall、prefix checkpoint）。整体合入等于重写 KVMem。

可尝试抽取的独立部分：

- 每请求 TTFT / transfer 观测；
- cache source preservation 的语义（admission 等待期间保留 checkpoint 所有权）；
- preemptive scheduling 的 admission permit 思路；
- 不引入上游的 `ContextCache` 类层次。

评估：只做“诊断/指标 + 语义修正”，不做 manager 替换。

## 7. 建议顺序

1. ✅ Jinja cache boundary（已完成，6 文件 +77/−3，构建与短请求验证通过）
2. ✅ Graph 校准 dense 最小闭环（已完成，5 文件；探针实测 115–143 MiB，下限 256 MiB）
3. ✅ bat 切到 q3blocks + 开图 + 77,824（free 457 MiB，8.7K decode 66.3–66.7）
4. MoE：方案 B，先补 CPU NVFP4 / block-FP8 / routed_swiglu 形状
5. qwen4-exp：等 MoE 内核稳定后再接 execution/program
6. Context：只取指标与 cache source 语义

## 8. 验证清单

- `ninfer-serve` 构建（CUDA 13.4 / sm_120a）
- 36,352 / 36,864 / 49,152 开图启动
- 8.7K / 62K prompt decode + 接受率
- prefix cache：同 prompt 二次请求 `cache_n`
- 长跑：显存 retained 不持续增长

## 9. 128K prompt 实测（2026-10-25）

条件：520,402 字符 ≈ 137,465 token；B/R 61,440/16,384 = 77,824；开图 + 自动校准；nvfp4 KV；temp 0；thinking off；max_tokens 256。

| 模型 | draft | prefill tok/s | decode tok/s | 接受率 | wall |
|---|---:|---:|---:|---:|---:|
| q3blocks | 5 | 395.2 | **51.2** | 164/452 = 36.3% | 353.1s |
| q3blocks | 7 | 361.9 | 46.1 | 171/575 = 29.7% | 385.6s |
| q4blocks | 5 | 367.9 | 49.2 | 162/462 = 35.1% | 379.1s |
| q4blocks | 7 | 400.7 | 45.9 | 168/596 = 28.2% | 348.9s |
| nvfp4full | 5 | 389.7 | 50.1 | 165/447 = 36.9% | 358.1s |
| nvfp4full | 7 | 387.9 | 44.9 | 169/589 = 28.7% | 360.2s |

结论：

- 128K 档 **draft 5 全面优于 draft 7**（decode +8~14%，接受率高 6~9 个点）；
- 三个模型差距在测量噪声内：q3 51.2 > nvfp4 50.1 > q4 49.2（draft 5）；
- 128K prompt 的 decode 比 8.7K 的 66 tok/s 低约 23%，主要来自长上下文与 KVMem 驱逐/回收；
- 137K prompt 超过 77,824 的 KV 窗口，本次是 **KVMem 窗口召回**下的速度，不是 128K 全驻留；
- 128K 全驻留需要 B+R ≈ 131,072，约多 0.93 GiB KV；q3 开图极限 86,016（free 313 MiB），**16 GB 下放不下 128K 全驻留**；
- q4/d5 在 77,824 下 free 仅 267.7 MiB，长跑余量最小，不建议作为 128K 档首选。

推荐 128K prompt：**q3blocks + draft 5 + B/R 77,824**（当前 bat）。
