# 2026-10-25 更新日志

> 已汇总到 [CHANGELOG.md](CHANGELOG.md)；本文保留原始细节与过程数据。

## 概要

| 类型 | 内容 |
|---|---|
| 修复 | Jinja 模板 trim 后 cache boundary 丢失 |
| 功能 | CUDA Graph 启动校准（探针实测，替代拓扑盲预留） |
| 配置 | DFlash2 启动器切到 q3blocks + 自动校准 |
| 回退 | KVMem 重选回到满窗口 100%，放弃 80% 水位（issue #6） |
| 修复 | CUDA Graph 探针读数抖动（高于 256 MiB 时复测） |
| 测试 | 128K / 256K 速度、NIAH 召回、q3/q4/nvfp4、MTP vs DFlash2 |
| 归档 | 本地 git 裸仓库 + 二进制备份 |

详细数据见 `docs/integration-20261025.md`。

---

## 1. Jinja 模板 trim 的 cache boundary

**问题**：`trim_blocks` / `lstrip_blocks` / `{%- ... -%}` 会把一段 source 折叠成 0 字节，落在折叠区的 cache marker 无法解析，prefix cache 命中下降。

**改动**（6 文件，+77/−3）：

- `third_party/llama-jinja/jinja/string.{h,cpp}`
  - `string_part::collapsed_source_end`
  - `append_boundary()`
  - `cut_bytes(begin, end, collapse_removed)`
  - `strip()` 走 `cut_bytes(..., true)`
- `third_party/llama-jinja/jinja/value.cpp`
  - `tojson` 只在唯一非折叠 part 上保留 origin
- `src/text/jinja.{h,cpp}`
  - `TemplateBoundaryMapping`
  - `TemplateOutput::boundary_mappings`
- `src/models/qwen3_5/frontend/prompt_layout.cpp`
  - `source_boundary()` 查询折叠边界

**效果**：模板边界附近的 prefix cache marker 可以唯一解析，不再因为 trim 丢失。

---

## 2. CUDA Graph 启动校准

**问题**：DFlash2 的 graph allowance 按 topology 盲预留，约 1 GiB；实际 Graph 只有 90–143 MiB，导致 KV 少给、长上下文启动失败。

**实现**（5 文件，+81/−1）：

- `src/models/qwen3_5/program/program.{h,cpp}`
  - `measure_cuda_graph_bytes()`：构造最小探针 Program，读 `cuda_graph_measured_bytes`
- `src/models/qwen3_5/program/planning/startup.{h,cpp}`
  - `make_graph_probe_plan()`：DFlash2 用 KVMem 窗口作逻辑容量，物理页取 `page_count(window)`，绕过容量曲线最小值
- `src/runtime/engine/model_instance.cpp`
  - 最终容量规划前跑探针
  - allowance = `max(measured, 256 MiB)`
  - 探针失败回退盲预留
  - `--cuda-graph-allowance-mib` 仍优先
  - `--no-cuda-graph` 跳过探针

**效果**：

```text
probe measured 90.0–143.5 MiB
planning allowance 256 MiB
```

省下约 700–800 MiB，KV 容量从 36,352 提升到 77,824（q3blocks 开图），q2blocks 可到 69,632。

**修掉的问题**：

1. 探针关闭 context cache 触发 host budget 校验 → 保留配置的 cache；
2. `finalize(1)` 报 “outside the target capacity curve” → 新增 probe plan 绕过曲线；
3. 1 页 + max_context 报 “Paged KV physical pages are below logical capacity” → 用 KVMem 窗口作探针逻辑容量；
4. WDDM 把 free delta 夹到 0，校准被静默跳过 → 返回 `optional`，成功但 measured=0 时仍用 256 MiB 下限。

**探针读数抖动（0013，2026-10-26 凌晨）**：连续 6 次启动同一配置，实测值为
0.0 / 83.3 / 90.0 / 117.3 / 135.6 / **288.6** MiB。dedicated delta 会把 WDDM 的
换出与延迟加载模块一起算进去。288.6 那次把规划值抬到 288.6 MiB，B/R=4.12 的配置因此
启动失败：

```text
FATAL server failed during startup | requested Engine runtime reservation requires
2403183872 bytes, but only 2353864704 bytes are available for runtime capacity
```

现在首次读数高于 256 MiB 时会复测一次并取较小值；256 MiB 下限不变，常规配置的规划不变。

---

## 3. DFlash2 启动器

当前 `workers/ninfer/start-kvmem-ninfer-dflash.bat`：

```bat
set "MODEL=...q3blocks.ninfer"
--kvmem-budget 36480 --kvmem-gen-reserve 16384 --kvmem-host-mib 12288
--kv-dtype nvfp4
--spec dflash2 --draft-tokens 5 --ngram-draft-tokens 31
--default-reasoning-effort medium --default-thinking-budget 16384
```

- B/R = 36,480 / 16,384 = 52,864
- 开图 + 自动校准
- 不再需要 `--cuda-graph-allowance-mib`
- 不再需要 `--no-cuda-graph`

测试过的更高配置：

| 配置 | 总容量 | free | 结果 |
|---|---:|---:|---|
| B=61,440 / R=16,384 | 77,824 | 457 MiB | ✅ 8.7K decode 66.3–66.7 |
| B=65,536 / R=16,384 | 81,920 | 385 MiB | ✅ |
| B=69,632 / R=16,384 | 86,016 | 313 MiB | ✅ 极限 |
| B=73,728 / R=16,384 | 90,112 | — | ❌ 差 13 MiB |

---

## 3.5 KVMem 重选回到 100%（issue #6，2026-10-26）

**背景**：0002 把重选水位从满窗口 `W` 提前到 `W - W/5`，为的是让交换和 decode 重叠。
当 `B > 4R` 时，压缩目标 `B` 本身就高于 80% 水位，于是每个 256-token chunk 都触发一次
重选、每次只前进一个 chunk，就是 issue #6 的 thrash。

**试过并放弃的方案**：

| 方案 | 结果 |
|---|---|
| 增长门限 v1（基线 `compact_position`） | 253K prefill 崩溃 `position precedes compact KV append frontier` |
| 增长门限 v2（基线 `selected_tokens` / 0） | 同样崩溃 |
| 条件回退（`B > 4R` 用 100%，否则 80%） | 3/3 通过，但保留两套触发点 |

**最终改动**（0012，`kvmem_window.cpp`）：任何 B/R 都只在 `compact_position > W` 时重选。

```cpp
const bool reselecting =
    force_selection || compact_position(sequence, end) > kvmem_window_tokens;
```

- 两次重选之间至少增长一个 R，不会 no-op；
- 不新增状态，不碰 checkpoint 路径；
- 启动时不再有 `B > 4R` 的 WARN（它只为 80% 水位存在）。

**验证**：

| 场景 | 80% 水位 | 100% |
|---|---:|---:|
| 61K prompt，B/R=4.15，reselect | 55 | **0** |
| 61K prompt，B/R=4.15，wall | 82.2 s | **75.8 s** |
| 5 轮会话，B/R=4.15，reselect | 759 | 未重跑 |
| 256K NIAH，B/R=4.15 | 3/3（条件回退版） | 同代码路径 |

**代价**：平均可见窗口回到 `(W + B) / 2`，B/R=2.23 时约大 12%；10-19 记录的 256K
端到端 −12% 不再有。换来单一触发点和 `B > 4R` 不 thrash。

---

## 4. 测试与基准

### 8.7K prompt（256 token 输出）

| 配置 | decode | 接受率 |
|---|---:|---:|
| q3blocks + draft 5 + B/R 77,824 | 66.3–66.7 tok/s | 43.6% |

### 128K prompt（137,465 token，256 token 输出）

| 模型 | draft | prefill | decode | 接受率 | wall |
|---|---:|---:|---:|---:|---:|
| q3blocks | 5 | 395.2 | 51.2 | 36.3% | 353.1s |
| q3blocks | 7 | 361.9 | 46.1 | 29.7% | 385.6s |
| q4blocks | 5 | 367.9 | 49.2 | 35.1% | 379.1s |
| q4blocks | 7 | 400.7 | 45.9 | 28.2% | 348.9s |
| nvfp4full | 5 | 389.7 | 50.1 | 36.9% | 358.1s |
| nvfp4full | 7 | 387.9 | 44.9 | 28.7% | 360.2s |

结论：draft 5 全面优于 draft 7；三个模型 decode 差距在噪声内。

### 128K NIAH（q3 + nvfp4 KV + draft 5）

| 深度 | prompt_n | prefill | decode | 命中 |
|---:|---:|---:|---:|---:|
| 5% | 137,476 | 400.6 | 38.0 | ✅ BLUEBIRD-42 |
| 50% | 137,471 | 399.6 | 37.9 | ✅ BLUEBIRD-42 |
| 95% | 137,471 | 400.5 | 37.7 | ✅ BLUEBIRD-42 |

5% 已超出 77,824 设备窗口，命中说明 KVMem 召回有效。

### 256K NIAH（q3 + nvfp4 KV + draft 5，3 needle 一次 prefill）

| 深度 | station | code | 命中 |
|---:|---|---|:-:|
| 5% | 2047 | BLUEBIRD-42 | ✅ |
| 50% | 4093 | MAPLE-91 | ✅ |
| 95% | 8191 | REDWOOD-77 | ✅ |

```text
prompt_n 253,104
prefill  370.5 tok/s
wall     683.9s
decode   48.8 tok/s（19 token）
接受率   12/35 = 34.3%
```

### 256K q3 vs nvfp4

| 模型 | prefill | decode | 接受率 | wall |
|---|---:|---:|---:|---:|
| q3blocks | 370.5 | 48.8 | 34.3% | 683.9s |
| nvfp4full | 333.7 | 46.4 | 34.3% | 759.2s |

q3 快 11% prefill / 5% decode，召回都是 3/3。

### 256K MTP 3 vs DFlash2 5（同一 253K/256-token 报告 prompt）

| 后端 | prefill | decode | 接受率 | wall |
|---|---:|---:|---:|---:|
| MTP draft 3 | 369.1 | 52.4 | 51.2% | 690.7s |
| DFlash2 draft 5 | 369.7 | 50.5 | 33.5% | 689.8s |

MTP 接受率高 17.7 个点，decode 快 3.8%；prefill/wall 基本相同。

---

## 5. Git / 备份

本地提交（主仓库，引擎已 vendor）：

```text
183d3af  fix(startup): confirm an above-floor CUDA graph probe with a second probe
731bbbc  fix(kvmem): reselect at the full window again, dropping the 80% high-water
60de31d  vendor: fold the ninfer engine into the main repository
7ab2762  docs(patches): add 0012 KVMem reselect growth gate
```

- `backends/ninfer` 不再是子模块，GitHub 主仓库可以直接浏览引擎源码；
- 子模块 fork `melis2023/ninfer-all` 的 `dflash2-kvmem-fixes-20261018` 已强制更新为
  `4be30d4f`（撤销了会崩溃的增长门限），只作历史备份。

标签：`local-20261025`

本地裸仓库：

```text
I:\model\backup\local-git\kvmem-llama.cpp.git
I:\model\backup\local-git\ninfer.git
```

二进制备份：

```text
I:\model\backup\20261025-post-integration\
```

已 push GitHub：`https://github.com/melis2023/ninfer-kvmem-actkv`。

---

## 6. 未完成 / 限制

- 80% 水位已回退：平均可见窗口回到 `(W + B) / 2`，10-19 的 256K −12% 收益不再有；
- 128K / 256K 不是全驻留，仍是 77,824 窗口 + Host 召回；
- 256K prompt 的 prefill 会跑两遍（第二遍是 hidden replay，不计 token），报告速度约为
  瞬时速度的一半；这是既有行为，待单独排查；
- DFlash2 接受率随上下文下降：8.7K 43.6% → 128K 36.3% → 256K 33.5%；
- MTP 3 在 256K 更稳（51.2%），但未验证短上下文/创意任务；
- NVFP4 drafter 的 32-token A16 前向在 256K prefill 下慢于 q3 GGUF blocks；
- 未实现 DFlash2 自适应 draft 长度；
- 未移植上游 `19367582` 的“一次纠偏重试”；
- qwen4-exp / MoE / Context 重写未合入：上游 891 个冲突文件，本地已有自己的 Flash-Next 路径。
