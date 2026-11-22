# KVMem 召回路径优化 —— 改动前全量备份记录

日期：2026-10-19
目的：在实施召回优化（O4 选择器 comparator / O1 分数缓存 / O3 统计读回同步）之前，
对**整个仓库**做逐字节完整备份，保证任何一步都能整体回滚。

## 1. 备份快照

| 项 | 值 |
|---|---|
| 源目录 | `I:\llama\kvmem-llama.cpp-multi-backend-framework` |
| 备份目录 | `I:\llama\_backup\kvmem-llama.cpp-multi-backend-framework-20261019-074016` |
| 开始时间 | 2026-10-19 07:40:16 |
| 文件数 | 6955 |
| 总字节 | 5,977,892,087（约 5.567 GiB） |
| robocopy 结果 | 目录 1088 / 文件 6955 / 失败 0 / 不匹配 0，exit=1（正常，有复制） |
| 校验 | SRC/DST 文件数一致、字节数一致（diff=0）；5 个关键文件 SHA256 全部 OK |

备份为**全量镜像**，包含 `.git`、`build-backends`（2.97 GB）、`workers`（1.84 GB，含部署 exe 与日志），
即备份时刻的完整可运行状态。

## 2. 备份命令（可复现）

```powershell
$src = 'I:\llama\kvmem-llama.cpp-multi-backend-framework'
$dst = "I:\llama\_backup\kvmem-llama.cpp-multi-backend-framework-$(Get-Date -Format yyyyMMdd-HHmmss)"
robocopy $src $dst /MIR /COPY:DAT /DCOPY:DAT /R:1 /W:1 /XJ /NFL /NDL /NP /MT:16
```

## 3. 抽样 SHA256（备份时状态）

| 文件 | 结果 |
|---|---|
| `kvmem\src\host\mean_key_retrieval.cpp` | OK |
| `kvmem\src\host\kvmem_store.cpp` | OK |
| `kvmem\include\kvmem\kvmem_store.hpp` | OK |
| `backends\ninfer\src\models\qwen3_5\program\storage\memory_statistics.cpp` | OK |
| `backends\ninfer\src\models\qwen3_5\program\storage\kvmem_window.cpp` | OK |

## 4. 回滚方法

任选其一：

1. **整体还原（推荐，最干净）**
   ```powershell
   $dst = 'I:\llama\_backup\kvmem-llama.cpp-multi-backend-framework-20261019-074016'
   $src = 'I:\llama\kvmem-llama.cpp-multi-backend-framework'
   robocopy $dst $src /MIR /COPY:DAT /DCOPY:DAT /R:1 /W:1 /XJ /NFL /NDL /NP /MT:16
   ```
   注意 `/MIR` 会删除源目录中备份之后新增的文件，若要保留新产物改用 `/E` 覆盖式还原。

2. **只还原被改的源码文件**（保留 build 目录与日志）
   ```powershell
   $dst = 'I:\llama\_backup\kvmem-llama.cpp-multi-backend-framework-20261019-074016'
   Copy-Item "$dst\kvmem\src\host\*.cpp"  'I:\llama\kvmem-llama.cpp-multi-backend-framework\kvmem\src\host\' -Force
   Copy-Item "$dst\kvmem\include\kvmem\*.hpp" 'I:\llama\kvmem-llama.cpp-multi-backend-framework\kvmem\include\kvmem\' -Force
   Copy-Item "$dst\backends\ninfer\src\models\qwen3_5\program\storage\*.cpp" 'I:\llama\kvmem-llama.cpp-multi-backend-framework\backends\ninfer\src\models\qwen3_5\program\storage\' -Force
   ```

3. **git 兜底**：备份目录内含 `.git`，可直接在备份目录里 `git status` / `git diff` 比对改动。

## 5. 本次计划修改的文件（逐步补充）

| 步骤 | 文件 | 性质 |
|---|---|---|
| O4 | `kvmem\src\host\kvmem_store.cpp`（`pick_topk_ungrouped` 的 `nth_element` comparator） | 纯 host，低风险 |
| O1 | `kvmem\src\host\mean_key_retrieval.cpp` + `kvmem\include\kvmem\mean_key_retrieval.hpp` + `kvmem\src\host\kvmem_store.cpp(hpp)` | 分数缓存/增量，中风险 |
| O3 | `backends\ninfer\src\models\qwen3_5\program\storage\memory_statistics.cpp` | 同步策略，评估后再定 |

每步完成后在此追加「实际改动 + 行号 + 实测数据」，未完成的步骤标注原因。

## 6. 改动记录

### 6.1 O4：选择器 comparator 预取分数（已实施，已编译部署）

文件：`kvmem\src\host\kvmem_store.cpp`

- `pick_topk_ungrouped()` 的 `take_top`：原来 comparator 每次比较都 `score_fn(blocks_[a])`，
  现在先一次性填 `vector<pair<double,uint32_t>>`，comparator 只读连续内存。
- 同函数的最终填充分支：原来每次比较调用 `selection_score(a)` **两次**（一次比较、一次取值），
  现在同样预取。
- 语义完全保持：分数降序、tie-break 用 id 降序（与原实现一致）。
- 附带：`#include <utility>`。

### 6.2 探针：`KVMEM_RESELECT` 计时日志（已实施）

文件：`backends\ninfer\...\program\storage\kvmem_window.cpp`

每次重选打印一行：

```
KVMEM_RESELECT logical=<n> blocks=<n> score_us=<n> select_us=<n> plan_us|transfer_us=<n> selected=<n>
```

- `score_us`：`mean_key_retrieval_scores` 纯评分
- `select_us`：`set_retrieval_scores` + `pick_topk_blocks`
- `transfer_us`：`MemorySession::prepare + advance(含 spill/restore) + publish`
- 仅在真正重选时输出（低频：64k prompt 全程 2 次）

### 6.3 实测（部署后首次运行）

模型：`I:\model\ninfer\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.ninfer`
参数：`--kvmem-budget 36864 --kvmem-gen-reserve 16384 --spec dflash2 --draft-tokens 4`
payload：`sweep.json`（prompt 59336，greedy，max_tokens 200）

| 指标 | 值 |
|---|---|
| draft accepted | 124 / 296 = **41.9%** |
| decode | 54.5 tok/s |
| prefill | 382.9 tok/s |
| 总耗时 | 158.8 s（prompt_ms 154976） |

重选分解：

| 轮次 | logical | blocks | score_us | select_us | transfer_us | selected |
|---|---|---|---|---|---|---|
| 1 | 53248 | 832 | 0 | 0 | 1,413,697 | 576 |
| 2 | 53248 | 832 | 40,712 | 38 | 943,949 | 576 |

结论：

- **评分占 4.3%，选择占 0.004%，传输占 95%+。**
- 第一轮 `score_us=0` 是因为 `query_complete=false`（query 区间未 prefill 完），走纯 recent/mandatory 路径，无召回。
- 每次重选传输的是**整个窗口 576 页**（compaction 后 `destructive_truncate(0)` 全量重映射），
  约 1.7 GB H2D，实测约 1.8 GB/s，故 0.94–1.4 s/次。
- 因此 **O1（评分缓存）/ O2（评分向量化）不做**：上限收益 40 ms/次，且需要跨库缓存失效设计。
- **O3（统计 D2H 同步）也不做**：`key_sums` 是 per-chunk 临时缓冲（`bucket = at/64 - begin/64`），
  不可延迟读取；且 prefill 每 chunk 末尾本就有同步，收益近似为零。

### 6.4 待定：传输层优化（未实施）

真实瓶颈是"每次重选全量重建 576 页"。可选的下一步：

1. 接受现状：R=16384 已把 64k prompt 的重选压到 2 次，总传输 2.36 s（占 155 s 的 1.5%）。
2. 改 compaction 为增量（只换出/换入滑动部分，约 256 页）：理论省 56%，但涉及 `compact_position`
   与 `destructive_truncate` 语义，风险高。
3. 参数权衡：R=8192 换 B+8192（召回 +19%），代价是重选次数翻倍（每次仍 ~0.94 s）。

### 6.5 A/B 实测：这次改动到底提升多少

同模型（Swift-1.5 MTP）、同参数、同 `sweep.json`（prompt 59336 / greedy / 200 tokens），
只换 exe：

| 指标 | 旧 exe（F1–F5，hash F7CA…） | 新 exe（+O4+探针，hash 338D…） | 差异 |
|---|---|---|---|
| draft accepted | 124 / 296 = **41.9%** | 124 / 296 = **41.9%** | **0** |
| decode | 55.83 tok/s | 54.47 tok/s | −2.4%（噪声内） |
| prefill | 386.27 tok/s | 382.87 tok/s | −0.9%（噪声内） |
| prompt_ms | 153,611 | 154,977 | +1.4 s |
| 墙钟 | 157.3 s | 158.8 s | +1.5 s |

结论：

- **接受率没有任何变化**（选出的块集合完全相同，符合预期）。
- **速度没有提升**；1–2% 的差异属于单次运行噪声（两次重选传输本身就有 ±0.2 s 波动）。
- 选择器优化只占到重选耗时的 0.004%，分母是 944 ms 的页传输，因此 O4 的价值是"消除一处无效开销"，
  不是提速。
- 本轮真正的产出是**定位**：把"评分/选择/传输"从猜测变成 40 ms / 0.04 ms / 944 ms 三个数字，
  从而否决了 O1/O2/O3 三个原计划。

## 7. 传输层实测（第二轮：P1 假设被推翻）

### 7.1 本轮改动

- `kvmem_window.cpp`：`start_spill` 增加 `KVMEM_SPILL` 分段计时；`start_restore` 增加
  `KVMEM_RESTORE` 分段计时与 `runs` 统计。
- `paged_kv_cache.{h,cpp}`：新增诊断接口 `DeviceKVPagePool::contiguous_runs()`（只读，不参与拷贝路径）。
- `start_spill` 的清零从"整窗 memset"改为"只清 partial 页"（完整页会被 `copy_to_host_records`
  整页覆盖，原清零 1.33 GB 属纯浪费）。

### 7.2 实测（64k，R=16384，B=36864，两轮重选）

| 阶段 | 页数 | 数据量 | 耗时 | 有效带宽 |
|---|---|---|---|---|
| spill D2H（设备 → staging） | 832 | 1,326 MB | 306 ms | 4.55 GB/s |
| spill 写 archive（staging → 页面记录） | 832 | 1,326 MB | 70 ms | 19.9 GB/s |
| restore 读 archive | 576 | 918 MB | 46 ms | 21.0 GB/s |
| restore H2D（staging → 设备） | 576 | 918 MB | 256 ms | 3.76 GB/s |
| 单次重选合计 | 1,408 | 2,244 MB | 857–1,002 ms | — |

关键读数：

- `KVMEM_RESTORE … runs=1` → 576 个目标页**物理完全连续**，没有碎片。
- `sync_us` 仅 7–19 µs，时间全部耗在 `cudaMemcpy2DAsync` 的提交阶段：拷贝按
  (plane, head) 拆成约 136 次，每次宽度只有 12 KB、源 pitch 1.59 MiB，链路利用率上不去。
- 宿主内存带宽 20 GB/s，PCIe 实测约 4 GB/s；链路为 **Gen4 x8**（`nvidia-smi` current width=8，
  理论约 13 GB/s），仍有约 3 倍空间，但受设备 HeadMajor 布局限制。

### 7.3 结论

- **P1（连续页分配）不成立**：`DeviceKVPagePool` 本就优先取连续 run，实测 `runs=1`。
- **P2（多 copy stream）基本无效**：拷贝已同步化，瓶颈是单次 2D 描述符的宽度，不是并发度。
- **P3（spill‖restore 重叠）不可行**：两阶段共用同一 staging，且 restore 必须先 truncate。
- 真正的浪费是**数据量**：窗口只滑 256 页（16,384 token），却每次全量 spill 832 页 + 全量
  restore 576 页，约 60% 是重复搬运。理论最小 512 页（816 MB），可把 0.9 s 压到约 0.25 s。
- 要拿到这 4 倍收益，必须让保留页的物理映射在窗口滑动后保持不变（ring-buffer 窗口），
  属架构改动；当前配置下总收益 <1.5%，风险与收益不成比例 → **本轮不实施**。

### 7.4 本轮净收益

- 接受率与速度无回归：124/296、decode 56.9 tok/s、prefill 393 tok/s。
- 清零修正每次重选省约 40 ms；其余改动为诊断能力，不改变语义。
- 参数侧维持 R=16384（已把 64k 的重选次数压到最低）。若要换召回质量，可试
  R=8192 / B=45056，代价为 6–7 次重选 × 约 0.86 s ≈ 5.5 s（约 3.5%）。

### 7.5 配置 A/B：B=36864/R=16384 vs B=45056/R=8192

同一 exe、同一 payload，只改预算配比：

| 指标 | B=36864 / R=16384 | B=45056 / R=8192 |
|---|---|---|
| draft accepted | **124 / 296 = 41.9%** | 117 / 325 = 36.0% |
| decode | **56.89 tok/s** | 52.69 tok/s |
| prefill | 393.13 tok/s | 397.99 tok/s |
| prompt_ms | 150,933 | 149,091 |
| 重选 restore 页数 | 576 | 704 |
| 单次 transfer | 857–1,002 ms | 937–1,084 ms |

结论：

- 加大 B **降低**接受率 5.9 个百分点、decode 降 7%：多保留的旧页挤掉了对 draft 更有效的近期上下文。
- prefill 的 1% 提升属噪声，而 restore 页数从 576 涨到 704，传输反而更贵。
- **维持 B=36864 / R=16384** 不变。
### 7.6 复现性（同配置第三次跑）

固定 B=36864 / R=16384，同一 exe、同一 payload，连续三次：

| 指标 | 第 1 次 | 第 2 次 | 第 3 次 |
|---|---|---|---|
| draft accepted | 124/296 = 41.9% | 124/296 = 41.9% | 124/296 = 41.9% |
| decode | 56.89 tok/s | 56.85 tok/s | 55.75 tok/s |
| prefill | 393.13 tok/s | 391.97 tok/s | 376.10 tok/s |
| prompt_ms | 150,933 | 151,379 | 157,767 |
| 墙钟 | 153.0 s | 154.6 s | 161.5 s |
| 重选 1 transfer | 1,002 ms | 1,004 ms | 1,015 ms |
| 重选 2 transfer | 857 ms | 902 ms | 945 ms |

结论：

- **接受率逐位一致**（greedy + 固定窗口，选块集合确定），可作为 A/B 的硬指标。
- prefill 与墙钟存在 ±3–5% 的运行间抖动（第 3 次最慢），单次跑不足以判定 1–2% 的差异。
- 传输耗时稳定在 0.9–1.0 s/次，其中 `restore submit` 三次为 245/246/249 ms，
  波动 <2%，说明瓶颈是确定性的链路带宽，不是调度噪声。

## 8. 带宽利用率：从 26% 修到 96%

### 8.1 独立基准（同机、同卡，RTX 5060 Ti / Gen4 x8）

自写探针 `kv_bw_probe.cu`（918 MiB，1 次/64 次/8 次分块、单/双 stream、2D strided/contiguous）：

| 形态 | 耗时 | 带宽 |
|---|---|---|
| 1 × linear 918 MiB（H2D） | 68.28 ms | **14.10 GB/s** |
| 64 × linear（每 14 MiB） | 68.40 ms | 14.07 GB/s |
| 2 streams × linear | 68.34 ms | 14.09 GB/s |
| 64 × 2D strided（576×26112，srcPitch 1.59 MiB） | 68.44 ms | **14.07 GB/s** |
| 64 × 2D contiguous | 68.40 ms | 14.07 GB/s |
| D2H 1 × linear | 67.38 ms | 14.29 GB/s |
| D2H 64 × 2D strided | 67.36 ms | 14.29 GB/s |
| **H2D 与 D2H 并发（不同引擎）** | 67.82 ms | **28.39 GB/s（双向合计）** |

结论：**2D strided 形状本身不是问题，多 stream 也没有收益**，单链路天花板约 14 GB/s；
但 H2D/D2H 走不同 copy engine，双向并发可到 28 GB/s。

### 8.2 服务内的真实瓶颈：`record_groups` 语义用错

`KVMEM_PLANE` 探针（`KVMEM_PLANE_TIMING=1` 时逐 plane 打印）显示每次重选产生
**192,896 次 `count=1` 的 plane 拷贝**，即每页每 plane 单独搬 26 KiB。

原因：`copy_host_run` 的合并条件里有一项 `groups[next] == groups[next-1]`，
而 `record_groups` 的契约（`paged_kv_cache.h:305-310`）是**“每条记录所在的分配区”**。
`kvmem_window.cpp` 传入的是 `spill_ids` / `selected`，即**块 id**；块 id 不连续
（sink、recent、topk 混排），合并条件恒不成立，576 页被拆成 576 次单页传输。

设备侧其实是连续的（`runs=1`），宿主记录也来自同一块 pinned staging
（`staging + i * page_bytes`，步长恒定），所以这个拆分纯属误伤。

### 8.3 修复

4 处调用把 `spill_ids`/`selected` 改为空 groups（等价于“同一分配区，不做额外限制”）：

- `start_spill`：text / MTP 两次 `copy_to_host_records`
- `start_restore`：text / MTP 两次 `copy_from_host_records`

空 groups 语义与 `verify` 分支原本的写法一致；合并仍受
“设备页连续 + 宿主步长恒定 + 步长在 `[page_stride, maxCopyPitch]`”三重约束，安全性不变。

### 8.4 效果

| | 修复前 | 修复后 |
|---|---|---|
| spill 提交 `copy_us` | 306 ms | **0.36 ms** |
| spill 实际 DMA `sync_us` | ≈0 | **97 ms → 13.7 GB/s** |
| restore 提交 `submit_us` | 242 ms | **0.34 ms** |
| restore 实际 DMA `sync_us` | ≈0 | **68 ms → 13.5 GB/s** |
| 单次重选 `transfer_us` | 857–1,002 ms | **463–584 ms** |
| 链路利用率 | **26%** | **96%** |

A/B：接受率 `124/296 = 41.9%` 不变（逐位一致），decode 54.4 tok/s、prefill 374.4 tok/s，
均在噪声内——64k 只有 2 次重选，省下的约 0.9 s 占总时长 0.6%，端到端不可见。

正确性：带 `--kvmem-verify-transfers` 回放，两次重选均 `payload_exact=1 state_exact=1`，
无异常，接受率一致。

### 8.5 剩余空间（带宽已经到顶，只能少搬）

修复后的单次重选 463 ms 构成：D2H 97 + H2D 68 + 写 archive 72 + 读 archive 47 + 其它约 180。
两条方向各自都已贴着 14 GB/s，继续提速只剩三条路：

1. **少搬 60%（2,244 MB → 816 MB）**：窗口只滑 256 页，却全量 spill 832 + 全量 restore 576。
   需要保留页映射稳定（ring-buffer），约 0.46 s → 0.22 s。
2. **双向并发**：spill 的 D2H 与 restore 的 H2D 走不同引擎，实测合计 28.4 GB/s；
   但当前流程被 `destructive_truncate(0)` 串行化，需要分块流水化。
3. **去掉一次宿主中转**：D2H 落在 staging → 再 memcpy 进 archive（72 ms）→ restore 再读回（47 ms）。
   若 archive 本身是 pinned 且与设备同序，可省约 120 ms。

三条都不涉及"更快的内存拷贝"，只涉及"少搬/重叠/换布局"——这也是
`llama.cpp-adaptive-kv-streaming` 真正的做法：它的宿主历史与设备同序，所以每次都是
一条线性 `cudaMemcpyAsync`，并且用 ready/consumed fence 把搬运与计算叠起来。

## 9. 256k 召回实测与 A/B

### 9.1 payload 与配置

- `workers/ninfer/sweep256k.json`：1,199,982 字符，实测 `prompt_n = 247,064`。
- 配置不变：`B=36864 / R=16384`（窗口 53,248）、`--max-context 262144`、`--spec dflash2 --draft-tokens 4`。
- 同一 payload、同一模型、同一参数，只换 exe。

### 9.2 触发情况

- 24 次真实重选：每 16,384 token 滑动一次，且每个滑动点触发 **2 次**
  （一次 `score_us=0` 的强制选择 + 一次带分数的召回）。
- 页数：spill 平均 322 页（增量已生效），restore 固定 576 页。

### 9.3 数据

| 指标 | 对照 exe（块 id 作 groups） | 修复 exe（空 groups） | 差异 |
|---|---|---|---|
| prompt_n | 247,064 | 247,064 | — |
| prefill | 340.78 tok/s | 343.68 tok/s | +0.85% |
| prompt_ms | 724,991 | 718,879 | −6,112 ms |
| 墙钟 | 729.8 s | 723.7 s | −6.1 s |
| decode | 45.84 tok/s | 46.04 tok/s | +0.4% |
| draft 接受 | 112/343 = 32.7% | 112/343 = 32.7% | 0 |
| 重选 transfer 合计 | 12,374 ms | **6,144 ms** | **−50.3%** |
| 单次 transfer 均值 | 495 ms | **246 ms** | −50.3% |
| 单次 transfer 最大 | 994 ms | 595 ms | −40% |

DMA 明细（24 次合计）：

| 阶段 | 页数 | 数据量 | 对照 | 修复 | 带宽 |
|---|---|---|---|---|---|
| spill D2H | 7,722 | 12.9 GB | 3,028 ms 提交 | **908 ms DMA** | 4.26 → **14.2 GB/s** |
| restore H2D | 13,824 | 23.1 GB | 5,952 ms 提交 | **1,637 ms DMA** | 3.88 → **14.1 GB/s** |
| restore 读 archive | — | 23.1 GB | 1,139 ms | 1,209 ms | 19 GB/s（宿主） |
| spill 写 archive | — | 12.9 GB | 681 ms | 707 ms | 19 GB/s（宿主） |

### 9.4 结论

- DMA 阶段提升 3.4–3.5 倍，两条方向都贴着链路（14.1–14.2 GB/s），再快的拷贝不存在。
- 端到端只省 6.1 s（0.84%），因为 24 次重选里另有 1.9 s 宿主 memcpy、约 1.7 s
  plan/remap/事件开销，本轮没动。
- 接受率逐位一致（112/343），确认合并只改搬法、不改内容。

### 9.5 顺带发现（未处理）

每个窗口滑动点触发 **2 次** 完整重选：先一次 `score_us=0` 的强制选择，再一次带分数的召回，
传输整份做了两遍。若能合并为一次，256k 下还能再省约 3 s、64k 下省约 0.26 s。

## 10. 整体速度：时间账与可加性验证

### 10.1 端到端（同 payload、同参数、只换 exe）

| 指标 | 对照 | 修复 | 差异 |
|---|---|---|---|
| 端到端墙钟 | 729.8 s | **723.7 s** | **−6.1 s / −0.84%** |
| prefill `prompt_ms` | 724,991 ms | 718,879 ms | −6,112 ms |
| decode `predicted_ms` | 4,342 ms | 4,322 ms | −20 ms（噪声） |
| 总吞吐（prompt+200） | 338.8 tok/s | **341.7 tok/s** | **+0.86%** |
| TTFT | 725.0 s | 718.9 s | −6.1 s |
| 每 1k prompt token | 2,934 ms | **2,910 ms** | −0.83% |

### 10.2 可加性验证：省的时间真的进了墙钟

| 项目 | 对照 | 修复 | 差 |
|---|---|---|---|
| 传输合计（代码内计时，非推算） | 12,374 ms | 6,144 ms | **−6,230 ms** |
| `prompt_ms` 差 | — | — | −6,112 ms |
| 两者吻合度 | — | — | **98.1%** |
| 纯计算 = `prompt_ms` − 传输 | 712,617 ms | 712,735 ms | +118 ms（+0.017%） |
| 纯计算吞吐 | 346.71 tok/s | 346.65 tok/s | −0.02% |

结论：传输省下的 6.23 s 有 98% 落进墙钟；纯计算路径零回归（0.017% 在计时噪声内），
证明这次改动只影响搬法、不影响计算。

### 10.3 传输在总时长中的占比

| 场景 | 重选次数 | 对照传输 | 修复传输 | 占 prefill（对照 → 修复） |
|---|---|---|---|---|
| 64k（prefill 151 s） | 2 | 1.9 s | ~1.05 s | 1.26% → 0.70% |
| 256k（prefill 725 s） | 24 | 12.4 s | **6.1 s** | **1.71% → 0.85%** |

64k 的端到端墙钟不可用：三次对照跑为 153.0/154.6/161.5 s（±5%），
修复后又遇系统负载，单次跑无法分辨 0.9 s 的差异；只有 256k 的同 payload A/B 是干净的。

### 10.4 为什么整体只有 0.84%

因为传输本来只占 1.71%，修完降到 0.85%，剩下 99.15% 是计算、采样与调度。
这条路径的价值不在「省 6 秒」，而在：

1. 消除了每次重选 192,896 次 `count=1` 的小拷贝（CPU 提交与驱动队列压力）；
2. 传输耗时从「随页数线性膨胀」变成「贴链路带宽」，窗口越小、重选越密，收益越大；
3. 为下一步（去宿主中转、双向并发）建立了可测量的基线。

### 10.5 修复后 6.1 s 传输的构成

| 阶段 | 耗时 | 说明 |
|---|---|---|
| restore H2D | 1,637 ms | 14.1 GB/s，**已到链路极限** |
| 读 archive → staging | 1,209 ms | 宿主 memcpy，19 GB/s |
| spill D2H | 908 ms | 14.2 GB/s，**已到链路极限** |
| 写 staging → archive | 707 ms | 宿主 memcpy，19 GB/s |
| 其它（plan/remap/事件/小 spill） | ~1,700 ms | |
| **合计** | **6,144 ms** | |

宿主 memcpy 合计 1,916 ms（占 31%），是下一步最容易砍掉的一段。

## 11. 新旧逐次对比（256k，24 次重选）

### 11.1 代码差异

唯一改动是 4 处调用点的第 3 个实参：

```diff
  p.text_kv_pages->physical_pool().copy_to_host_records(sources, spill_records,
-     spill_ids, backend.layout, p.compute_streams);
+     {}, backend.layout, p.compute_streams);

  p.backend_kv_pages->physical_pool().copy_to_host_records(mtp_sources, mtp_records,
-     spill_ids, *backend.mtp_layout, p.compute_streams);
+     {}, *backend.mtp_layout, p.compute_streams);

- pool.copy_from_host_records(restore_records, selected, destinations, backend.layout, p.compute_streams);
+ pool.copy_from_host_records(restore_records, {}, destinations, backend.layout, p.compute_streams);

  p.backend_kv_pages->physical_pool().copy_from_host_records(mtp_records,
-     selected, mtp_destinations, *backend.mtp_layout, p.compute_streams);
+     {}, mtp_destinations, *backend.mtp_layout, p.compute_streams);
```

### 11.2 稳态单次重选的时间线（第 2 次，`logical=69632`，spill 256 页 / restore 576 页）

| 阶段 | 旧（块 id） | 新（空 groups） | 变化 |
|---|---|---|---|
| spill D2H（427.8 MB） | 107.8 ms = **3.97 GB/s** | 29.8 ms = **14.36 GB/s** | **−72.4%** |
| spill 写 archive | 21.8 ms | 23.2 ms | +1.4 ms |
| restore 读 archive | 45.9 ms | 48.4 ms | +2.5 ms |
| restore H2D（962.6 MB） | 234.2 ms = **4.11 GB/s** | 68.2 ms = **14.11 GB/s** | **−70.9%** |
| 小计 | 409.7 ms | 169.6 ms | −58.6% |
| 其它（plan/remap/事件） | 57.6 ms | 59.8 ms | +2.2 ms |
| **`transfer_us`** | **467.3 ms** | **229.4 ms** | **−50.9%** |

冷启动那次（第 1 次，spill 832 页 = 1,390 MB）：旧 D2H 344.4 ms = 4.04 GB/s，
新 109.7 ms = **12.68 GB/s**，`transfer_us` 994.1 → 609.6 ms。

两次拷贝的提交耗时对照：旧 `copy_us=107,817`（提交即阻塞）、`sync_us=95`；
新 `copy_us=366`（提交）、`sync_us=29,849`（真正 DMA）——耗时从驱动队列搬到了 GPU 拷引擎。

### 11.3 24 次逐次对比

| # | 上下文 | spill 页 | restore 页 | 旧 transfer | 新 transfer | 省 | 倍数 |
|---|---|---|---|---|---|---|---|
| 1 | 53,248 | 833 | 576 | 994.1 ms | 609.6 ms | 384.5 | 1.63× |
| 2 | 69,632 | 256 | 576 | 467.3 | 229.4 | 237.9 | 2.04× |
| 3 | 86,016 | 256 | 576 | 461.1 | 224.8 | 236.3 | 2.05× |
| 4 | 102,400 | 256 | 576 | 473.8 | 228.2 | 245.6 | 2.08× |
| 5 | 118,784 | 256 | 576 | 461.9 | 229.1 | 232.8 | 2.02× |
| 6 | 135,168 | 256 | 576 | 467.8 | 227.0 | 240.8 | 2.06× |
| 7 | 151,552 | 256 | 576 | 471.9 | 226.9 | 245.0 | 2.08× |
| 8 | 167,936 | 256 | 576 | 474.8 | 231.2 | 243.6 | 2.05× |
| 9 | 184,320 | 256 | 576 | 510.3 | 235.7 | 274.6 | 2.17× |
| 10 | 200,704 | 256 | 576 | 475.9 | 224.6 | 251.3 | 2.12× |
| 11 | 217,088 | 256 | 576 | 483.6 | 224.4 | 259.2 | 2.16× |
| 12 | 233,472 | 256 | 576 | 479.8 | 229.6 | 250.2 | 2.09× |
| 13 | 53,248 | 832 | 576 | 876.2 | 497.7 | 378.5 | 1.76× |
| 14 | 69,632 | 256 | 576 | 476.2 | 239.0 | 237.2 | 1.99× |
| 15 | 86,016 | 256 | 576 | 479.7 | 223.3 | 256.4 | 2.15× |
| 16 | 102,400 | 256 | 576 | 473.1 | 227.4 | 245.7 | 2.08× |
| 17 | 118,784 | 256 | 576 | 476.8 | 223.0 | 253.8 | 2.14× |
| 18 | 135,168 | 256 | 576 | 469.9 | 231.1 | 238.8 | 2.03× |
| 19 | 151,552 | 256 | 576 | 484.2 | 224.6 | 259.6 | 2.16× |
| 20 | 167,936 | 256 | 576 | 475.8 | 224.3 | 251.5 | 2.12× |
| 21 | 184,320 | 256 | 576 | 492.5 | 242.4 | 250.1 | 2.03× |
| 22 | 200,704 | 256 | 576 | 489.0 | 230.0 | 259.0 | 2.13× |
| 23 | 217,088 | 256 | 576 | 462.5 | 230.7 | 231.8 | 2.00× |
| 24 | 233,472 | 256 | 576 | 492.7 | 223.6 | 269.1 | 2.20× |
| | **合计** | 7,721 | 13,824 | **12,371 ms** | **6,138 ms** | **6,233** | **2.02×** |

配套汇总：

| 项目 | 旧 | 新 | 变化 |
|---|---|---|---|
| 提交耗时（`copy_us` + `submit_us`） | 8,821 ms | **18 ms** | −99.8% |
| 实际 DMA（`sync_us`） | 5 ms | **2,502 ms** | +2,497 ms |
| 写 archive | 642 ms | 666 ms | +24 ms |
| 读 archive | 1,139 ms | 1,182 ms | +43 ms |
| 单次均值 | 515 ms | **256 ms** | −50.3% |

旧版把 8.8 s 的传输时间记在“提交”上（逐页同步），GPU 拷贝引擎几乎空闲（DMA 仅 5 ms）；
新版提交只要 18 ms，真正干活的是拷贝引擎（2.5 s @ 14 GB/s）。

### 11.4 端到端（修复版跑两次）

| 指标 | 旧（1 次） | 新第 1 次 | 新第 2 次 | 新均值 | 变化 |
|---|---|---|---|---|---|
| 墙钟 | 729.8 s | 723.7 s | 724.6 s | **724.2 s** | **−5.6 s / −0.77%** |
| `prompt_ms` | 724,991 | 718,879 | 719,869 | **719,374** | −5,617 ms |
| prefill | 340.78 tok/s | 343.68 | 343.21 | **343.4** | **+0.78%** |
| decode | 45.84 tok/s | 46.04 | 46.14 | 46.09 | +0.5%（噪声） |
| 总吞吐 | 338.8 tok/s | 341.7 | 341.5 | **341.6** | **+0.79%** |
| draft 接受 | 112/343 | 112/343 | 112/343 | 112/343 | 0 |
| 纯计算（扣传输） | 712,620 ms / 346.70 tok/s | 712,735 / 346.65 | 713,731 / 346.16 | 713,233 / 346.41 | −0.08%（噪声） |

两次独立跑的新版墙钟只差 0.9 s（0.12%），而新旧差 5.6 s——差异是信号、不是抖动。



