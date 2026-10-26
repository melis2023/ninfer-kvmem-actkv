# 修改与基准总览

> 最后更新：2026-10-26。本文件汇总历次修补的**原因、改动和验证数据**；专题细节仍在
> `docs/` 下的原始记录里，关键结论以本文件为准。

## 0. 当前状态

| 项 | 值 |
|---|---|
| 主力工件 | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp-q3km.ninfer` |
| 目标模型 | IQ3_S GGUF 块（131 个 `gguf_iq3_s`） |
| 草稿 | q3km 直通：`attention/output`、`mlp/down` = Q4_K×10；feature/selector = Q3_K×2 |
| 草稿后端 | DFlash2，`draft 4`，`ngram 0`（agent 场景可用 15） |
| KV | `nvfp4`，B/R = 36,480 / 16,384（W=52,864） |
| KV 显存 | 1.61–1.67 GiB（826 页） |
| CUDA Graph | 开；探针实测 60–157 MiB，规划 256 MiB |
| 驱逐 | 满窗口 100% |
| 16 GB 卡启动 free | 780–955 MiB |
| 日常 decode | 鹈鹕 74–90 tok/s；agent 9.8K 89 tok/s；代码 128K 87 tok/s |

---

## 1. 修复分类索引

| 类别 | 日期 | 提交 | 结论 |
|---|---|---|---|
| KVMem 召回 | 10-20 | — | ActKV LRFU 替代 EMA：LongMemEval 54/60 vs 50/60 |
| KVMem 传输 | 10-20 | — | 差异 restore：64K 163 ms → 33–51 ms |
| KVMem 驱逐 | 10-26 | `c8cc858` | 重选回到满窗口 100%：61K 55 次 → 0 次 |
| KVMem 打分 | 10-22 | — | layer-wise 已回退：慢 3–5%，召回 53/60 |
| DFlash2 × KVMem | 10-18 | — | envelope/floor/slot 修复：越窗接受率 1.5% → 42% |
| DFlash2 floor | 10-19 | — | floor 64 → 2048：越窗接受率 +3.7–5.4 点 |
| DFlash2 floor | 10-26 | `0d66244` | 三处公式统一，防御性修正，实测无变化 |
| DFlash2 bypass | 10-26 | `eddada7` `fbf79f2` | 按期望收益判定：误关草稿 0 次 |
| ngram 仲裁 | 10-26 | `10d1da2` | copy 只在收益 > 1.3× 草稿时触发 |
| CUDA Graph | 10-25 | `b4d8096` | 探针校准，省 700–800 MiB，不再需要 `--no-cuda-graph` |
| CUDA Graph | 10-26 | `591537b` | 读数 >256 MiB 时复测取小 |
| NVFP4 scale | 10-26 | `798988c` | Blackwell 保留值 `0x7F` 映射为 0（仅编译验证） |
| 模板缓存 | 10-25 | `46a0d0d` | Jinja trim 后 cache boundary 不再丢失 |
| 平台 | 10-22 | — | sm70/sm75 源码支持（未运行验证） |

---

## 2. KVMem 召回与驱逐

### 2.1 ActKV LRFU（2026-10-20）

**问题**：EMA 是加权平均，偶尔命中但单次分高的块活不久。

**改动**：

```text
hit : KS_t = λ·KS_{t-1} + score
miss: KS_t = λ·KS_{t-1}
```

新块从 0 开始，首次命中时点亮。

**验证**：

| 场景 | EMA | ActKV LRFU |
|---|---:|---:|
| 64K prefill | 422.4 tok/s | 399.7 tok/s |
| 64K decode | 49.3 tok/s | 46.3 tok/s |
| 64K 接受率 | 51.3% | 51.3% |
| 256K NIAH 33 针 | 33/33 | 33/33 |
| 256K 跨工具 | 14/14 | 14/14 |
| 256K 端到端 | 594.8 s | 646.9 s |
| LongMemEval 54 题 | 50/60 | **54/60** |
| Agent 220K 召回 | — | **30/30** |

LongMemEval 分档：128K 20/20、192K 18/20、240K 16/20。最佳工作区间 128K–192K。

### 2.2 增量 restore（2026-10-20）

**问题**：每次重选 `truncate(0)` 后全量重映射，64K 一次 163 ms。

**改动**：

- 选中页按 id 稳定排序，同一逻辑页保持物理槽位；
- 只对变化槽位做 H2D，只读变化 host 记录；
- 变化超过一半自动回退全量。

**验证（64K，5 次重选）**：

| 轮次 | changed | restore 页数 | read | sync | 总耗时 |
|---|---:|---:|---:|---:|---:|
| 全量基线 | 560 | 576 | 69 ms | 94 ms | 163 ms |
| 增量 | 124 | 124 | 14 ms | 20 ms | **33 ms** |
| 增量 | 164 | 164 | 20 ms | 27 ms | **47 ms** |
| 增量 | 181 | 181 | 21 ms | 29 ms | **51 ms** |

256K 时 `changed` 中位数 495/576，自动回退全量；端到端 594.8 s。

### 2.3 重选回到满窗口 100%（2026-10-26，issue #6）

**问题**：80% 水位在 B > 4R 时，压缩目标 B 本身高于水位，每个 256-token chunk
都触发一次空转重选。

**改动**（`kvmem_window.cpp`）：

```cpp
const bool reselecting =
    force_selection || compact_position(sequence, end) > kvmem_window_tokens;
```

**验证（61K prompt，B/R=4.15）**：

| | 80% 水位 | 100% |
|---|---:|---:|
| reselect 次数 | 55 | **0** |
| wall | 82.2 s | **75.8 s** |
| 256K NIAH | 3/3 | 3/3 |

**代价**：平均可见窗口回到 `(W+B)/2`；10-19 记录的 256K −12% 不再有。

### 2.4 layer-wise 打分（2026-10-22，已回退）

| 场景 | ActKV | layer-wise | 差异 |
|---|---:|---:|---:|
| 64K prefill | 443.4 | 428.2 | −3.4% |
| 64K decode | 46.7 | 44.5 | −4.7% |
| LongMemEval | 54/60 | 53/60 | −1 |

结论：额外层权重没有带来召回收益，反而增加开销，已回退。

### 2.5 块大小固定 64

`kPagedKVPageSize` 同时是 fast prompt 内核的 K 方向 tile：

```cpp
static_assert(kCausalPromptI8FastBc == kPagedKVPageSize);
```

改成 32 需要重写 MMA 迭代和共享内存预算；48 不满足页表 `position >> 6` 的 2 的幂要求。
**块大小不可调**。

---

## 3. DFlash2 × KVMem

### 3.1 envelope / floor / slot fork（2026-10-18）

**现象**：上下文超过约 2048 token 后，DFlash2 草稿接受率从 32–55% 掉到 0–1.5%。

**根因**：

1. `dflash_envelopes` 用 `max_context` 短路，envelope 语义错误；
2. KVMem 检索缝合后，draft ring 读到被驱逐的历史，floor 掩码不足；
3. state fork 后槽位未跟随；
4. dflash 轮缺少 `prepare_window`，窗口满时 entitlement 溢出。

**验证（greedy）**：

| prompt token | 修复前接受率 | 修复后 | decode |
|---:|---:|---:|---:|
| 2,366 | 0.4% | **43.2%** | 30.9 → **72.7** tok/s |
| 3,510 | 1.5% | **39.6%** | 31.3 → **72.8** tok/s |
| 4,654 | 1.5% | **41.8%** | 31.8 → **75.7** tok/s |
| 131K 缝合窗口 | 0/36 | 正常 | — |

### 3.2 floor 保留量 64 → 2048（2026-10-19）

**问题**：`recent_tokens = 64`，但 DFlash2 草稿环容量是 2048；每次重选后草稿只剩
3.1% 的 conditioning 上下文，接受率被钉在 30–39%。

**改动**：floor 改为 `max(recent × 64, masked_draft_ring_capacity())`。

**验证（64K 越窗）**：

| 配置 | 修复前 | 修复后 |
|---|---:|---:|
| draft 4 接受率 | 30.1% | **33.8%** |
| draft 3 接受率 | 35.4% | **40.7%** |
| t2 紧随 64K（draft 4） | 34.3% | **39.7%** |

### 3.3 floor 三处统一（2026-10-26，`0d66244`）

重选、query restore、checkpoint recovery 三处 floor 公式不一致；后两处只写 `recent`，
会隐藏 2048 窗口的大部分。统一为：

```cpp
max(recent * 64, masked_draft_ring_capacity())
```

**实测：没有变化。** 128K 总结 prompt、q3km draft 4 修复前后都是 135/477 = 28.3%，
decode 43.8–45.8 tok/s。属于防御性一致性修正。

### 3.4 bypass 按期望收益判定（2026-10-26，`eddada7` `fbf79f2`）

**问题**：原逻辑“连续 6 轮 accepted == 0 就关草稿”，d4 下即使 p1=0.4 每轮期望仍有
约 1.4 token，6 连零在几十轮内必然出现。结果是思考阶段被误关，decode 掉到约 40 tok/s，
还要等最多 64 轮才恢复。

**改动**：

- 用 `MtpAdaptiveSignal` 的按位置 EMA 计算 `expected_tokens`；
- `expected < 1.15` 连续 6 轮才 bypass；
- 探测间隔 64 → 约 10 轮；探针未回本立即重新 bypass。

**验证（鹈鹕，draft 4，ngram 0）**：

| | 修复前 | 修复后 |
|---|---:|---:|
| `DFLASH_BYPASS` 事件 | 1–3 | **0** |
| decode | 73.2 tok/s | 73.1 tok/s |
| 接受率 | 58.6% | 55.3%¹ |

¹ 两次输出长度不同（19,723 vs 14,357 token），接受率有采样噪声。
代码任务 128K：接受率 82.1% 不变，bypass 0。

### 3.5 ngram copy 按收益仲裁（2026-10-26，`10d1da2`）

**问题**：只要有 ≥12 token 的 ngram 匹配就整轮 copy，DFlash 被跳过，验证宽度固定 32 列；
短匹配或近期 copy 一直不准时，反而比草稿慢。

**改动**（`NgramCopySignal`）：

```text
expected_copy = (accepted / drafted) × match_length
copy 条件     = expected_copy > neural_expected × 1.3
```

- 前 2 轮 copy 无条件放行，用于建立样本；
- 连续 8 轮拒绝后强制探测一轮；
- bypass 时 neural 基线按 1.0 计算；
- 图 family 不变：copy 仍用 `ngram_graphs`，拒绝回到 `dflash_graphs` / `mtp_graphs`。

**验证**：

| 场景 | ngram 31 | ngram 0 | copy 轮 |
|---|---:|---:|---|
| 鹈鹕 HTML（draft 4） | 74.7 tok/s | 72.1 tok/s | 84 轮，753/2,604 |
| 38 工具 agent（9.8K） | 89.2 tok/s | 87.8 tok/s | 2 轮，29/59 |

结论：ngram 31 不再拖慢 DFlash，两个场景都略快；copy 轮只保留真正有收益的匹配。

---

## 4. 启动与显存

### 4.1 CUDA Graph 探针校准（2026-10-25，`b4d8096`）

**问题**：按 topology 盲预留约 1 GiB，实际 Graph 只占 90–145 MiB，导致 KV 少给、
长上下文启动失败，只能 `--no-cuda-graph`。

**改动**：启动时用最小 KV 探针实测整套图，规划 `max(实测, 256 MiB)`；探针失败回退盲预留；
`--cuda-graph-allowance-mib` 仍优先；`--no-cuda-graph` 跳过探针。

**效果**：

- 省约 **700–800 MiB**；
- q3blocks 开图 KV 容量 36,352 → **77,824**；
- 开图 decode 比关图快约 **6%**；
- 不再需要 `--no-cuda-graph`。

### 4.2 探针读数复测（2026-10-26，`591537b`）

同一配置连续 6 次启动实测：0.0 / 83.3 / 90.0 / 117.3 / 135.6 / **288.6** MiB。
288.6 那次把规划值抬高到容量之外：

```text
FATAL | requested 2403183872 bytes, but only 2353864704 bytes are available
```

现在首次读数 > 256 MiB 时复测一次取较小值；256 MiB 下限不变。

### 4.3 NVFP4 保留 scale `0x7F`（2026-10-26，`798988c`）

存储的 UE4M3 权重 scale 用 `0x7F` 表示 0，软件展开路径正确处理；Blackwell 原生
`cvt.rn.bf16x2.e4m3x2` 会把它解成 NaN。在 native pair 路径加：

```cpp
if (scale == 0x7Fu) { scale = 0; }
```

NVFP4 KV codec 未动，与上游一致。**本地无 NVFP4 工件，仅编译验证。**

### 4.4 q3km 直通草稿（2026-10-26）

**改动**：DFlash2 草稿的线性权重按源 GGUF 块格式直接加载，不再统一重化成 Q8/NVFP4。
q3km 工件的实际格式：

| 张量 | 格式 | 数量 |
|---|---|---:|
| `attention/output`、`mlp/down` | `gguf_q4_k` | 10 |
| `feature_projection`、`selector/hidden_projection` | `gguf_q3_k` | 2 |
| conv、norm、codebook | BF16 | 其余 |

**验证（128K，nvfp4 KV，ngram 0）**：

| 草稿 | prefill | decode | 接受率 |
|---|---:|---:|---:|
| q3km 直通 | **904.8 tok/s** | **42.8 tok/s** | 23.4% |
| 混合 NVFP4 | 844.5 tok/s | 40.7 tok/s | 23.8% |
| q4km 直通 | 904.8 tok/s | 43.6 tok/s | 24.2% |

- 直通比混合 NVFP4 prefill 快 6.7%、decode 快 4.9%；
- q3 与 q4 在噪声内，q4 多占约 0.18 GB → 推荐 q3km；
- Q8 草稿（14.6 GB）在 B/R=36,480/16,384 下启动失败（差 184 MiB），需缩到 W=40,960。

---

## 5. 模板与前缀缓存

### Jinja trim 的 cache boundary（2026-10-25，`46a0d0d`）

`trim_blocks` / `lstrip_blocks` / `{%- -%}` 会把 source 折叠成 0 字节，落在折叠区的
cache marker 无法解析，prefix cache 命中下降。

改动：`string_part::collapsed_source_end` + `append_boundary()` + `cut_bytes(...)`；
`TemplateBoundaryMapping` 跟随折叠边界；`prompt_layout.cpp` 查询 `source_boundary()`。

效果：模板边界附近的 marker 可以唯一解析。

---

## 6. 平台移植

### sm70 / sm75（2026-10-22）

| 项 | 内容 |
|---|---|
| 新增文件 | 31 个（sm70 GEMM、sm75 MMA、`pre_sm80.cuh` / `sm70_emulation.cuh`） |
| 接入修改 | 8 处（CMake、`mma.cuh`、`memory.cuh`、sources、`QType`、2 个 codec） |
| GPU profile | 9 个（V100×2、TITAN RTX、2060S、2070、2070S、2080Ti、Quadro RTX 6000、T4） |
| ActKV | 未改动 |
| 验证 | sm120a 编译 0 错误；64K 427.1 / 45.5 / 48.1%；sm75/sm70 **未运行验证** |

---

## 7. 性能基准

### 7.1 128K 代码（148,774 token，补函数，256 token 输出）

| 后端 | prefill | decode | 接受率 | wall |
|---|---:|---:|---:|---:|
| DFlash2 + q3km，draft 4 | 826.5 | **87.1** | 82.1% (92/112) | 350.7 s |
| MTP + q3km，draft 3 | 826.5 | 72.2 | **86.9%** (86/99) | 349.5 s |

### 7.2 128K 总结（126,668 token，256 token 输出）

| 后端 | prefill | decode | 接受率 | wall |
|---|---:|---:|---:|---:|
| DFlash2 + q3km，draft 4 | 904.8 | 45.8 | 28.3% | 288.5 s |
| MTP + q3km，draft 3 | 904.8 | **47.7** | **40.8%** | 280.9 s |

### 7.3 256K（253,099 token，256 token 输出，两个启动器）

| 启动器 | 草稿 | KV | prefill（第一遍） | decode | 接受率 | wall |
|---|---|---|---:|---:|---:|---:|
| dflash | **draft 4** | nvfp4 | **854.8** | **50.5** | **37.8%** | 597.4 s |
| mtp | draft 3 adaptive | int8 | 803.3 | 47.5 | 40.7% | 630.9 s |

DFlash2 行为 2026-10-26 复测（q3km、`ngram 0`、当前二进制）：报告 prefill 427.9 tok/s
含隐藏 replay，第一遍按两倍折算；接受率 153/405。MTP 行仍是旧启动器数据。

### 7.4 鹈鹕 HTML draft 扫描（72 token prompt，ngram 0）

| draft | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|
| decode | 68.8 | **71.4** | 70.6 | 67.8 | 64.0 |
| 接受率 | **65.7%** | 54.9% | 48.4% | 42.4% | 38.1% |

### 7.5 短 prompt / agent

| 场景 | decode | 接受率 | 说明 |
|---|---:|---:|---|
| WebUI 鹈鹕（2 工具，thinking 209） | 96.0 | 76.7% | 输出 6,218 |
| harness（38 工具，thinking 4,826） | 68.2 | 51.9% | 输出 9,030 |
| agent 9.8K（ngram 0） | 87.8 | 70.8% | 输出 835 |
| agent 9.8K（ngram 31 仲裁） | 89.2 | 68.1% | copy 2 轮 |

### 7.6 KV 精度

| | `nvfp4` | `rk8v4` |
|---|---:|---:|
| 52,864 窗口 runtime | **1.62 GiB** | 1.99 GiB |
| 启动 free | **835.8 MiB** | 645.9 MiB |
| 鹈鹕 decode（draft 4） | **73.2 tok/s** | 68.6 tok/s |
| 同任务接受率 | **58.6%** | 51.3% |

`nvfp4` 显存最省，是 16 GB 卡默认；`rk8v4` 的 key 保持 INT8，理论检索更稳。

---

## 8. 召回汇总

| 测试 | 结果 |
|---|---|
| 256K NIAH，33 针 | 33/33（同轮 / 终轮严格 / 并集） |
| 256K NIAH，跨工具 `send_alert` | 14/14 |
| 256K NIAH，工具调用 | 80 |
| 128K NIAH，5% / 50% / 95% 深度 | 3/3 |
| 256K 三针（5% / 50% / 95%） | 3/3（`BLUEBIRD-42` / `MAPLE-91` / `REDWOOD-77`） |
| LongMemEval 54 题 | ActKV 54/60（128K 20/20、192K 18/20、240K 16/20） |
| Agent 220K | 30/30 逐轮工具召回，181 次工具调用 |

5% 深度已超出 52,864 设备窗口，命中说明 KVMem 的 Host 召回有效。

---

## 9. 已知限制

- **256K prefill 跑两遍**：第二遍是 hidden replay，不计 token，报告速度约为瞬时的一半；
- **DFlash2 接受率随上下文下降**（draft 5）：8.7K 43.6% → 128K 36.3% → 256K 33.5%；
  draft 4 在 128K 总结任务上是 28.3%；
- **DFlash2 没有自适应 draft 长度**；MTP 有 `--adaptive-mtp`；
- **Q8 草稿**只能配更小窗口（W=40,960），不适合 16 GB 主力配置；
- **本地无 NVFP4 目标工件**，`0x7F` 修复只做了编译验证；
- **上游未移植**：TP、query-memory、checkpoint-policy、monitor-events；
- **sm75/sm70** 只有源码与编译，未在真机运行。

---

## 10. 专题文档

| 文档 | 内容 |
|---|---|
| `docs/dflash2-kvmem-fixes-20261018.md` | DFlash2 × KVMem 首次修复（envelope/floor/slot） |
| `docs/dflash2-kvmem-audit-20261019.md` | 代码审查 F1–F6 与 floor 修复 |
| `docs/kvmem-recall-opt-20261019.md` | 召回传输优化与备份记录 |
| `docs/kvmem-reselect-high-water-20261019.md` | 80% 水位与 issue #6 的完整过程 |
| `docs/kvmem-actkv-lrfu-20261020.md` | ActKV LRFU 公式与 LongMemEval |
| `docs/kvmem-incremental-restore-20261020.md` | 差异 restore 实现与 64K 数据 |
| `docs/kvmem-layer-wise-scoring-20261022.md` | layer-wise 实验（已回退） |
| `docs/kvmem-sm75-sm70-port-20261022.md` | sm70/sm75 移植与块大小结论 |
| `docs/cuda-graph-calibration.md` | Graph 探针实现与启动数据 |
| `docs/dflash2-nvfp4-20261023.md` | NVFP4 草稿导入路径 |
| `docs/dflash2-passthrough-20261026.md` | q3km 直通、draft 扫描、floor/bypass 细节 |
| `docs/integration-20261025.md` | 上游整合评估与 128K/256K 数据 |
| `docs/CHANGELOG-20261025.md` | 10-25 当天的完整日志 |
