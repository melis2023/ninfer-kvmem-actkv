# ActKV 保留策略参数说明（中文）

适用范围：`workers\ninfer\start-kvmem-ninfer.bat` 命令行上的 ActKV 保留类开关。

英文对照：同一个 `.bat` 头部的注释块（编号 `(1)`–`(5)`）。本文件是它的中文版，另外补齐了
实施细节、取值边界、启用方式和 A/B 口径。

参数名与默认值由**共享的 kvmem core** 定义，所以 llama.cpp 与 ninfer 两个后端用的是同一套
参数（另见 `docs/multi-backend-differences.md`）。ninfer 侧只有两处差别会影响这里的读数：

- 页/块大小**固定 64 token**（llama.cpp 默认 128），所以预算、sink、recent 都向下取整到整页；
- 没有 `--kvmem-method`：查询完整时 ninfer 永远按 mean-K 打分，**LRFU 是这里唯一的保留策略旋钮**。

---

## 0. 一页速查

| 编号 | 参数 | 当前 bat | 作用 |
|---|---|---|---|
| (1) | `--kvmem-lrfu` | **开** | 用带时间衰减的 LRFU 保留分 KS 取代一次性注意力分 |
| (1b) | `--kvmem-lrfu-lambda F` | 默认 0.50 | 衰减系数：越大越偏"频率"，越小越偏"新近" |
| (2) | `--kvmem-score-agg` | **已删除** | GQA 头聚合恒为折叠 mean；传该参数（写 `max` 还是 `mean`）都是启动错误 |
| (3) | `--kvmem-conf-*` | **已删除** | 论文 §6 Alg. 2 自适应预算：实测换 15% 轮时、丢 23 根针，静态池也存不下省出的内存 |
| (4) | `--kvmem-protect-semantics` | **开** | 论文 §8.1：system 块 + 当前 user 任务**按需救援**，上限 = 窗口的 15% |
| (5) | `--kvmem-aav-protect` 等三个参数 | **已删除** | 论文 §5.1 的 AAV 是**诊断**（Fig. 6），不是保留策略；逐类改 λ 会冻结工作集、掉召回 |
| (6) | `--kvmem-query-union` | **已删除** | 论文 §5.1 的动作偏好：本 chat template 的 `tool_calls` 不产生输出区，探针从未激活 |
| — | `KVMEM_AAV_TRACE=1`（环境变量） | 未设 | 只统计不决策：打印访问模式分类与 `intermittent_dropped` |

---

## 1. (1) 动作导向的逐出 `--kvmem-lrfu`

每个块的保留分不再等于"这一轮它被注意到多少"，而是带时间衰减的累积量：

```
KS_t = λ · KS_{t-1} + (本轮 top-p 命中 ? 本轮检索分 : 0)
```

- λ 默认 **0.50**、top-p 默认 **0.90**，这是论文调好的值。
- λ → 0 退化为纯新近性（追最新 query）；λ → 1 退化为纯频率（工作集冻结）。
- 保留分在 0 处截断；一轮内两次打分不会把同一份池子衰减两次。

不开这个开关时，本节之外的机制里只有 §8.1（即 (4)）能单独用；(5) **必须**配它。

---

## 2. (2) GQA 头聚合 —— 已删除，不是"改了默认值"

论文 Alg. 3 第 8 步要求在一个 GQA 组内对 query head 取 **max**。这个实现被完整做过，然后在
40-needle / 256k 上实测，结果被自己否掉：

| 聚合 | 同轮召回 | 最终严格 | 工具调用 | 合计 |
|---|---|---|---|---|
| mean | 39/40 | 39/40 | 40 | 81 |
| max | 28/40 | 32/40 | 1 | 68 |

max 那次是**从第 6 轮开始逐轮衰减**（3/3 → 2/3 → 1/3），不是随机丢针，说明机制是累积性的：
逐 head 取 max 会被"恰好某个 head 尖起来"主导，令 LRFU 保留分变噪，动作关键块在后续轮次里
一个接一个丢掉自己的槽位。所以**整个参数被移除**，传 `--kvmem-score-agg`（写 `max` 还是
`mean` 都一样）现在是启动错误：

```
--kvmem-score-agg was removed: GQA aggregation is always the folded mean. The max variant
regressed 40-needle / 256k recall from 39/40 to 28/40, so there is nothing to configure
and the flag is gone without a stub.
```

聚合方式只有折叠均值一种，不再设开关（同时也便宜 g 倍）。

---

## 3. (3) 置信度驱动的预算 `--kvmem-conf-*` —— 已删除

论文 §6 Alg. 2 完整实现过：token 置信度 `T_i = -(1/k)·Σ log P_i` 从引擎已有的 top-k logprob
里取（k = 20，论文值），按 w = 64 token、stride 8 做 min-pool，取最低的 10% 拟合一条直线，
斜率为负时把预算乘上 `scale_up`（默认 1.75）。只在**每个压缩轮**检查一次，只增不减，上限
`--kvmem-budget`。

然后在 40-needle / 256k 上实测，被自己的数据否掉：

| 口径 | 同轮召回 | 轮时 |
|---|---|---|
| 关（B 恒为 36864） | **40/40** | 基准 |
| 开（B 从 8192 起步） | **17/40** | +约 15% |

根因是**静态页池**：物理池在启动时按 `B + R` 一次性分配，预算调小并不能把省下的显存"存起来"
留给后面几轮，所以这个机制在这里没有收益，只剩"早期轮次留得太少 → 丢掉后面的针"。整族
（`--kvmem-conf-budget / -scale-up / -cooldown / -pool / -stride / -bottom`）已删除，传任一都是
启动错误：

```
--kvmem-conf-budget was removed: the ActKV Sec. 6 adaptive budget traded ~15% of round wall
for a recall collapse (40/40 -> 17/40 at --kvmem-conf-budget 8192) and cannot save memory in a
static B+R pool. The selection budget is --kvmem-budget; drop the flag.
```

---

## 4. (4) 语义保护 `--kvmem-protect-semantics`（论文 §8.1，默认关，bat 开）

### 保护什么

打开后，`memory_semantic_pages()`（`backends/ninfer/src/models/qwen3_5/program/storage/kvmem_window.cpp:574`）
产出两段页：

| 段 | 来源 | 含义 |
|---|---|---|
| `[0, prompt.instruction_tokens)` | `leading_instruction_boundary()`（`frontend.cpp:560`）识别的前导 System/Developer 消息 | 规则块（"你是谁、怎么做事"） |
| `prompt.memory_query` 的 provenance span | 最后一条**真实** user 消息 | 当前任务 |

两个刻意的边界：

- 整段提示（`count == total`）不会被保护 —— 那等于让窗口完全没有压缩余地（同文件 `add_range` 里直接返回空）；
- 只覆盖**前导**的 system/developer 块，对话中间插入的 system 消息不在内。

### 怎么生效：按需救援，不是预先钉住

旧语义是"先占预算的硬钉住"（保护页并进 `mandatory` 再排名），**已废弃**。现行语义只有一个：

1. 这一轮**先按检索分正常选页**（`mandatory` 只含 sink + 图像 + recent，699-714 行）；
2. 选定后，看保护段里**哪些页被丢掉了**；
3. 把这些页按**检索分降序**救回，最多 `protect_cap` 个：
   `protect_cap = max(1, floor(窗口页数 × 0.15))`，其中 `kProtectRescueFraction = 0.15` 是
   **硬编码**，没有命令行参数也没有环境变量（737-739 行）；
4. 用"`mandatory` + 救回页"再选一次，得到最终窗口（同函数的 `pick_topk_blocks(narrowed)`）。

两个直接后果：

- 保护页如果检索本来就留下了，**一分预算都不花**（前导 instruction 前缀通常就是这种情况），
  不会挤掉检索分更高的页；
- 保护**永远吃不下超过 15% 的窗口**，"一条超长 user 消息钉死整个窗口"这种失败模式不可能再发生。

`0.25` 这个数来自 262k / 33 针 NIAH 实测（见下），不是拍脑袋。**代码里写的是 `0.15`**。
`protect_cap` 只在保护段本身很大时才起作用，两个尺度下的实测完全不同：

- **短提示**（41k / 57k / 72k token）：实测的保护规模（`protected=2`、`cap=86`）根本够不到上限，cap 只是"病态超长保护段"的
  安全阀；留到 0.15 是给检索分多留窗口，代价为零。
- **256k / 40 针长会话**（每轮灌 ~17k token）：它**会顶到上限** —— 33 个压缩轮里有 10 轮
  `rescued = cap = 86`，而那一轮请求救援的是 229–236 页（`0.25` 对应的上限是 144）。
  即便如此召回仍是 40/40，所以在这套负载上更紧的限额没有换来召回损失。

### 为什么需要

不开的话，只有"位置 sink 窗口 + recent 窗口"在护着它们。system prompt 一旦长过 sink，
尾巴落入可压缩区间，就会在 LRFU 对它失去兴趣时被丢掉 —— 而它对"当前要做什么"是决定性的。
论文的观点是：这类内容根本不该参与压缩。

### 代价与风险

同一台机器、同一份 262k / 33 针 NIAH（`niah33_262k.py`），三组对照：

| 口径 | 召回 | 工具轮 decode 中位 | 接受率 |
|---|---|---|---|
| 全保护（旧语义，等价 `frac = 1`） | 33/33 | 77.4 tok/s | 62.8% |
| 关闭保护（`frac = 0`） | 33/33 | 84.7 tok/s | 61.8% |
| **限额 0.25（实测口径）** | **33/33** | **90.4 tok/s** | 61.6% |

旧语义在长 user 消息上等于 `mandatory == selected == 576 页`（实测），窗口里没有一页留给打分，
代价约 9% decode 吞吐；改成按需救援 + 限额后代价消失，召回不变。上表的限额测的是 0.25，
代码现行 0.15（实测够不到 cap，见上，降到 0.15 零成本）。

剩下的风险只有一条：`protect_cap` 是**每轮**上限，保护段远大于 15% 窗口时，超出部分仍然按普通页
参与竞争（这是有意的 —— 保护是"救援"而不是"特权"）。图像页不同：它属于 `mandatory`，整块放不下
时选择阶段**直接报错**而不是悄悄丢内容 ——
`KVMem selection budget cannot retain complete image regions; increase the budget or reduce image resolution`（568 行）。

实践口径：长 system prompt + 视觉同时上时把 `--kvmem-budget` 留够；如果开始报预算类错误，
先抬预算，其次再考虑降 `--vision-max-merged` 或图像分辨率。

### 诊断：`KVMEM_PROTECT_TRACE=1`

设了以后每个压缩轮往 stderr 打一行（789-808 行），只打印、不改变任何决策：

```
KVMEM_PROTECT_TRACE logical=… window_pages=576 cap=86 protected=832 dropped_by_retrieval=256 rescued=86 span_tokens=[…,…) frac=0.15
KVMEM_PROTECT_TRACE rescued_pages=245(tok 15680-15744 s=1.234),…
```

`protected` = 保护段总页数；`dropped_by_retrieval` = 其中被检索丢掉的；`rescued` = 真正救回的（≤ `cap`）。
`rescued == dropped_by_retrieval` 说明没顶到上限；`rescued == cap` 说明保护段比限额大，被截掉的那部分
仍然会流失。想看保护到底有没有用，先看 `rescued`：它是 0 就说明保护段本来就没被丢，开关无差别。

### 与其它开关的关系

不依赖 (1)：`--kvmem-lrfu` 开或关都生效。与 AAV 诊断（§5）不互斥。做 A/B 时用上面的
`KVMEM_PROTECT_TRACE=1` 判定，比猜"是不是预算给少了"可靠。

---

## 5. AAV 访问模式分类 `KVMEM_AAV_TRACE=1`（论文 §5.1，**仅诊断**）

> **`--kvmem-aav-protect` / `--kvmem-aav-lambda-intermittent` / `--kvmem-aav-min-hits` 已删除，
> 传这三个参数是启动错误。**
> 论文 §5.1 的 AAV 是**分析**（Fig. 6）：把条目分成 cold-start / intermittent / continuous，
> 用来论证**一个全局 λ**。Algorithm 1 对 hit 和 miss 用的是同一个 λ，全文没有"按类别改 λ"。
> 代码曾实现过这个策略（intermittent 用 0.85 而不是 0.50），结果是把 λ 推向 1 ——
> 正是论文自己警告的 "frequency-dominated eviction may retain obsolete continuous entries"，
> 实测召回明显下降，故删除。AAV 只保留为诊断。

### 16 轮访问环与四分类

store 给每个块维护一条**访问环**：每轮一个 hit 位（这一轮它进没进保留集），窗口 16 轮。据此分类：

| 类 | 判据 |
|---|---|
| Silent | 出现以来一次没被保留 |
| Cold | 窗口内命中数不足 min-hits（常量 2） |
| Continuous | 观测到的每一轮都命中 |
| Intermittent | 有的轮命中、有的轮没命中（且命中数 ≥ min-hits） |

窗口只算**已观测**的 `min(轮数, 16)` 位，高位的补零不算 miss；窗口滚满后，miss 滚出窗口外，
分类会清回。分类只描述访问模式，**不参与任何保留决策**。

### 怎么采：`KVMEM_AAV_TRACE=1`

访问环**只在**设了 `KVMEM_AAV_TRACE=1` 时推进（测试里则直接设 `KvMemStoreConfig::aav_trace`）。
每轮往 stderr 打一行（`kvmem_store.cpp` 的 `record_access`）：

```
KVMEM_AAV_TRACE row=… frontier=… blocks=… selected=… cold=… intermittent=…
                intermittent_dropped=… continuous=… silent=… resurrected=…
                jaccard_prev=… jaccard_4=…
```

关键是 `intermittent_dropped`：**这一轮的选择丢掉了多少个 intermittent 块**。
它长期很小 → 负载里没有值得关注的断续块；它很大 → 说明当前 λ 正在丢掉会复活的条目，
该调的是论文自己的旋钮 `--kvmem-lruf-lambda`，而不是给某一类单独设 λ。

汇总：`python I:\model\256k-recall-test\aav_summary.py <server.err>`

---

## 6. 在 ninfer 启动器上怎么开

编辑规则（bat 头部也写了）：除最后一行外，每条参数行末尾必须是 ` ^`（前后各一个空格）；
最后一行不能有 `^`，否则会传一个空参数、服务在加载前就报 `unknown argument:`。

bat 里现在只开两个开关：`--kvmem-lrfu`（LRFU 保留分）与 `--kvmem-protect-semantics`
（§8.1：按需救援 system 块与当前 user 任务，上限 15% 窗口）。`--kvmem-query-union`、
`--kvmem-score-agg`、`--kvmem-conf-*`、`--kvmem-aav-*`、`--kvmem-action-region` 都已删除，
写进命令行会直接启动报错（错误信息里带删除原因和实测数据）。
AAV **没有命令行开关**，只有诊断环境变量（见上一节）；语义保护有 `KVMEM_PROTECT_TRACE=1`。

采数据模式：在 `cd /d "%BIN%"` 之后、exe 调用之前插一行

```bat
set "KVMEM_AAV_TRACE=1"
```

（脚本站开头有 `setlocal`，作用域只在本脚本内。）trace 走 stderr：直接看控制台，
或在参数末尾重定向 `2> "%~dp0aav-trace.log"`。

---

## 7. 取值与启动校验

| 参数 | 取值 | 违规时 |
|---|---|---|
| `--kvmem-lrfu-lambda` | (0,1) | `--kvmem-lrfu-lambda must be in (0, 1)` |
| `--kvmem-lrfu-top-frac` | (0,1] | 超出范围即启动错误（同类报错） |
| `--kvmem-lrfu-lambda` / `--kvmem-lrfu-top-frac`（任一显式给出） | 必须同时有 `--kvmem-lrfu` | `… require --kvmem-lrfu: without the LRFU keep score there is no decay or hit threshold to configure` |
| `--kvmem-sink-tokens` | 0（一页）或 ≥ 64 | `KVMem --kvmem-sink-tokens must be 0 (one page) or at least 64: a smaller non-zero value rounded to one page silently` |
| `--kvmem-sink-tokens` + `--kvmem-recent-tokens` | 取整到 64 页后不得超过 `--kvmem-budget` | `KVMem sink and recent tokens exceed --kvmem-budget after rounding to 64-token pages` |
| `--kvmem-score-agg` | 不接受任何值 | `was removed: GQA aggregation is always the folded mean…` |
| `--kvmem-conf-*`（六个） | 不接受 | `was removed: the ActKV Sec. 6 adaptive budget…` |
| `--kvmem-aav-protect` / `--kvmem-aav-lambda-intermittent` / `--kvmem-aav-min-hits` | 不接受 | `was removed: ActKV Sec. 5.1 AAV is a diagnostic, not a retention policy…` |
| `--kvmem-action-region` / `--kvmem-query-union` | 不接受 | `was removed: the action probe never activated…` |

全部在**参数解析期**就报错，且错误信息里带删除原因与实测数据。"被静默忽略"的取值已经不存在。

---

## 8. A/B 建议

(1) LRFU、(4) 语义保护、AAV 诊断互不排斥，但都作用在同一份 `--kvmem-budget` 上，所以一次只动
一个变量：

1. 先跑**基线**（40 针 / 256k 或 33 针 / 262k）→ 对齐历史结果
   （40 针基线的唯一例外 N26-063352 是已知单针，不计入）；
2. 再**只改一个**变量重跑，对比同轮 / 严格 / 重试三项；召回是最硬的指标，工具轮 decode 中位与
   接受率用来看代价（`niah33_262k.py` 同时给这两组数）；
3. 判定 (4) 有没有生效要用 `KVMEM_PROTECT_TRACE=1` 看 `rescued`，别只看召回 —— 保护段本来就没被
   丢时（`rescued = 0`）开关无差别，只看召回会得出"没用"的假结论；
4. AAV 不参与对比：它没有开关，只在你需要解释 λ 行为时用 `KVMEM_AAV_TRACE=1` 采数据，
   看 `intermittent_dropped` 的量级。

注意 `I:\model\256k-recall-test\run-recall-256k.ps1` 里的服务器参数是**另一份拷贝**（`$srvArgs`），
不会自动跟随 bat；做 A/B 必须两边同步，否则测的不是同一套配置。

---

## 9. 实现位置（维护用）

| 文件 | 内容 |
|---|---|
| `kvmem/include/kvmem/kvmem_store.hpp` | `KvAavClass`、`aav_trace`、`select_score`、`aav_class_of` |
| `kvmem/src/host/kvmem_store.cpp` | `aav_classify` / `kAavIntermittentMinHits`、`aav_class_of`、`record_access`、trace 行 |
| `kvmem/tests/kvmem_store_test.cpp` | 环形分类 / miss 滚出窗口后分类清除 |
| `backends/ninfer/include/ninfer/types.h` | `KvmemOptions` 新增字段（含 C 的 `protect_semantics`） |
| `backends/ninfer/src/serve/serve_options.cpp` | CLI 解析、范围校验、`--help` |
| `backends/ninfer/src/models/qwen3_5/program/storage/memory_statistics.cpp` | 把选项接进 `KvMemStoreConfig`（唯一构造点） |
| `backends/ninfer/src/models/qwen3_5/program/storage/kvmem_window.cpp` | `memory_semantic_pages()`(574)（§8.1）、`mandatory` 集合(700)、按需救援与 `kProtectRescueFraction`(737)、`KVMEM_PROTECT_TRACE`(789) |
| `backends/ninfer/src/models/qwen3_5/frontend/frontend.cpp` / `program/prefill.cpp` | `leading_instruction_boundary()`(512)、`instruction_tokens`、`protected_pages` |

λ 只有一个来源：`KvMemStoreConfig::lrfu_lambda`，被 `accumulate_lrfu`（已提交的更新）与
`set_retrieval_scores` 里的延迟发布共用 —— 所以 provisional 排名与最终提交用的是同一个 λ。
`--kvmem-aav-protect` 曾让 `lrfu_lambda_for()` 对 intermittent 块返回另一个 λ；该函数与策略已删除。

---

## 10. 状态

- **现行 bat 只开两个开关**：`--kvmem-lrfu`（λ、top-p 用默认值）与 `--kvmem-protect-semantics`
  （按需救援 + 15% 上限）。代码、CLI 接线、单元测试与 ninfer 二进制均已就位，`--help` 里有说明。
- **已删除**：`--kvmem-query-union` / `--kvmem-action-region`（动作探针从未激活）、
  `--kvmem-score-agg`（GQA 恒为折叠均值）、`--kvmem-conf-*`（静态池下无收益，实测掉 23 针召回）、
  `--kvmem-aav-*`（论文 §5.1 的 AAV 只是诊断）。传任一都是**启动错误**并给出原因与实测数据。
- `--kvmem-sink-tokens` 与 `--kvmem-lrfu-*` 不再有"静默失效"：非零 sink 小于一页、或 LRFU 参数
  没配 `--kvmem-lrfu`，都在启动期报错（见第 7 节）。
- AAV 保留为诊断：`KVMEM_AAV_TRACE=1` 打印分类行；语义保护用 `KVMEM_PROTECT_TRACE=1`；
  `I:\model\256k-recall-test\aav_summary.py` 已同步到新的行格式（此前的正则与实现脱节，匹配不到任何行）。
- `backends/patches/ninfer-kvmem.patch` 与 `backends/versions.json` 已按当前子模块工作树重生成，
  `scripts\prepare-backends.py --backend ninfer --check` 报
  `ninfer: 1383a50e4153 + ninfer-kvmem.patch ready`，标准路径 `build-backends.ps1` 不再卡在 prepare。
- base 已跟到上游 `1383a50e`（Turing/Volta RC，含 `sm_75`/`sm_70`），本地 ActKV 移植随之上线；
  Windows 构建所需的 FFmpeg/libcurl/zlib 现在固定在 `C:\vcpkg`（见 `docs/ninfer-server-build-record.md`）。
