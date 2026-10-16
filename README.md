------------------------------------------------------------------------------------------------------------------------------------

项目作用

基于 kvmem@（https://github.com/kvmem/kvmem-llama.cpp）与 ActKV 技术开发：

- **KVMem**：把长上下文稀疏化成固定窗口（`budget` 选取历史页 B + `gen_reserve` 追加工作区 R）+ Host/磁盘归档，用一个固定大小的注意力窗口换取 200K～262K 的可用上下文。
- **ActKV**（arXiv:2609.31395，中国科学技术大学提出）https://arxiv.org/pdf/2609.31395：修正长上下文下的“保留策略”——用动作导向的 LRFU keep score 与语义保护，弥补 KVMem 在 200K 以上上下文出现的幻觉、工具调用失效等问题。


项目仍处实验阶段，经本地多轮 256K 召回测试验证有效；actkv 技术属于中国科学技术大学其团队。本 git 项目仅提供参考，不作为标准。

------------------------------------------------------------------------------------------------------------------------------------

推荐模型下载

| 模型 | 下载地址 |
|---|---|
| **Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp**（本页测试所用的主模型，GGUF） | 魔搭：https://modelscope.cn/models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/file/view/master/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf |
| **Ternary-Bonsai-2-27B-NInfer-v3**（用 `--spec mtp` 加载，无 dflash，显存占用小于 12 G） | HuggingFace：https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3/tree/main |

> 魔搭上提供的是 GGUF，需用本仓库的 gguf→ninfer 转换工具转成 `.ninfer` 再使用（放到仓库根目录命名 `model.ninfer`，或修改启动脚本里的 `MODEL` 路径）。

------------------------------------------------------------------------------------------------------------------------------------

ActKV 的作用

| 参数 | 作用 |
|---|---|
| `--kvmem-lrfu` | 开启 ActKV 保留策略：块按时间衰减的 keep score 保留（`KS_t = λ·KS_{t-1} + 命中分`），而不是一次性注意力分 |
| `--kvmem-lrfu-lambda` / `--kvmem-lrfu-top-frac` | λ 与 top-p 命中阈值（默认 0.50 / 0.90，论文调优值）；**必须与 `--kvmem-lrfu` 同时给出**，单独给会启动报错 |
| `--kvmem-protect-semantics` | 语义保护：前段 system/developer 块 + 最后一条 user 消息的页**按需救援**，上限 = 窗口页数 × 25%（硬编码，无额外参数）——详见下节 |
| `--kvmem-budget` | ActKV 的**硬前置**：缺它启动报错（keep score 要在 KVMem 窗口内排序页） |

已删除的参数（传即**启动报错**，错误信息里写明原因）：`--kvmem-score-agg`（GQA 恒为折叠均值）、`--kvmem-conf-*`、`--kvmem-aav-protect`、`--kvmem-action-region`、`--kvmem-query-union`。ninfer 与 llama.cpp 的差异：页大小**固定 64 token**（llama.cpp 为 128），且没有 `--kvmem-method`。

### `--kvmem-protect-semantics` 具体介绍

- **保护什么**：前段 system/developer 块（指令前缀）与**最后一条真实 user 消息**——即这一轮真正要用到的页。
- **怎么保护（按需救援，不是先占预算）**：检索先照常按分数选出窗口页，然后**只把这轮被丢掉的保护页**按检索分从高到低救回来，上限 = **窗口页数 × 25%**（`kProtectRescueFraction`，硬编码：没有参数，也没有环境变量）。被检索本来就留下的保护页**一分预算都不花**，所以长指令前缀不会把高分页挤出去。
- **为什么定 25%**：旧语义（把保护区间一次性占满预算）在一条超长 user 消息上会把整个窗口钉死（当时配置 `--kvmem-budget 36864` 即 576 页全部 `mandatory == selected`），没有页面留给打分，实测损失约 **9% decode**；改成上限 25% 后，同样 262k/33 针仍是 33/33，且开销与关闭保护基本持平。
- **生效范围**：`--kvmem-lrfu` 开或关都生效——它只决定窗口里最终留哪些页，不改变选页与打分口径。
- **诊断**：`set KVMEM_PROTECT_TRACE=1`，每个压缩轮打印一行 `protected / dropped_by_retrieval / rescued` 及被救援的页列表，**不改变任何决策**；`rescued=0` 说明保护页本来就没被丢（这种情况开关开不开都一样）。

------------------------------------------------------------------------------------------------------------------------------------

最新启动参数

ninfer 后端（模型 `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.ninfer`）：

```
--port 18200 --model-id ninfer-27b
--max-context 262144 --kvmem-recent-tokens 1024
--prefill-chunk 256
--kvmem-budget 36864 --kvmem-gen-reserve 16384 --kvmem-host-mib 12288
--kvmem-lrfu --kvmem-protect-semantics
--kv-dtype int8 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--spec mtp --draft-tokens 4 --ngram-draft-tokens 31 --adaptive-mtp
--vision --vision-residency cpu --vision-max-merged 256
--default-reasoning-effort xhigh --default-thinking-budget 16384
```

- KV 池：`36864 + 16384 = 53,248 token`（832 页 × 64；`--kvmem-host-mib 12288` = 12 GiB 归档预算）
- `--vision --vision-residency cpu --vision-max-merged 256`：CPU 视觉需 256，GPU 可改为 1024
- dflash2 变体：`--kv-dtype rk8v4`、`--draft-tokens 5`、`--default-thinking-budget 53248`，其余同上

------------------------------------------------------------------------------------------------------------------------------------

支持的显卡架构（与上游同步：sm_70 / 75 / 80 / 86 / 89 / 120a）

本次把 ninfer 后端**同步到上游的 Turing/Volta 基线**（pin `1383a50e`），本分支的 KVMem/ActKV 补丁已整体移植到该基线（40 处冲突 + 编译期修正，`scripts/prepare-backends.py --backend ninfer --check` 通过）。允许的架构列表直接来自 `backends/ninfer/CMakeLists.txt`（源码注释：supports Ampere `sm_86` and early Ada `sm_89`, plus `sm_80`, `75`, `70`, `120a`）：

| 架构 | 代表显卡 | CUDA 要求 | 构建参数 |
|---|---|---|---|
| **`sm_89`（Ada）** | **RTX 40 系**（4060 / 4070 / 4080 / 4090）、L4 / L40 | CUDA 13.x 即可（也可用 12.x） | `-CudaArchitectures 89` |
| **`sm_86`（Ampere）** | **RTX 30 系**、A40 / A10 | CUDA 13.x 即可 | `-CudaArchitectures 86`（**上游源码默认**） |
| `sm_80`（Ampere GA100） | A100 / A30 | CUDA 13.x 即可 | `-CudaArchitectures 80` |
| **`sm_75`（Turing）** | **RTX 20 系**（2060 / 2070 / 2080 / 2080 Ti）、TITAN RTX、Quadro RTX | CUDA 13.x 即可 | `-CudaArchitectures 75` |
| **`sm_70`（Volta）** | **V100、TITAN V** | **必须 CUDA 12.8 / 12.9**（源码里 `sm_70 + CUDA ≥ 13.0` 直接 FATAL_ERROR） | `-CudaArchitectures 70` |
| `sm_120a`（Blackwell） | RTX 50 系（本页 256K 测试用的 RTX 5060 Ti） | **必须 CUDA ≥ 13.1**（12.8/12.9 会误编 sm_120a）；可选 `-DNINFER_SM120_NATIVE=ON` 走上游原生路由 | `-CudaArchitectures 120a` |

- 多显卡按**每架构独立构建、独立出包**（不做多 DLL 拆分，仍是单个 `infer-kvmem-server.exe`）：
  `scripts\windows\build-backends.ps1 -Backend ninfer -CudaArchitectures 89 -CudaPath <CUDA 路径> -Jobs 6`
- 非 120a 架构（70 / 75 / 80 / 86 / 89）走同一套 `mma.sync` 兼容路径；`sm_70` / `sm_75` 在 CMake 里另有专门分支（Turing/Volta 的能力差异）。
- 目前**只有 `sm_120a` 有本机实卡数据**：本页 256K 召回测试即 RTX 5060 Ti；其余架构属"可构建 + 上游编译通过级"，上游自述**手头没有 RTX 20 / TITAN RTX / V100 实卡**，RTX 30/40 系等其他卡请自行实卡验证。

------------------------------------------------------------------------------------------------------------------------------------

前端启动器（kvmem-launcher）

位置：**仓库根目录 `kvmem-launcher/`**（`launcher.py` + `web/index.html` + `config.json` + `verify.py` + `pick.py` + `start-launcher.bat`）——纯本地、零依赖的 Web 控制台，用来配置并管理 `workers\ninfer\infer-kvmem-server.exe`。

| 项 | 说明 |
|---|---|
| 启动 | 双击 `start-launcher.bat`，或 `python launcher.py`；`--no-browser` 不自动开浏览器，`--port 18301` 换端口 |
| 地址 | 默认 <http://127.0.0.1:18300>（只监听本机回环；后端只依赖 Python 3 标准库） |
| 页面 | **01 服务控制**（启动/停止服务、显示进程与端口）· **02 三页页签**（`启动参数` / `对话` / `日志`，同一张面板互斥显示，页签状态记在 `localStorage`）· **03 预设管理**（保存/切换预设，落在同目录 `config.json`）· **04 实时指标** |
| 04 指标 | 顶部横排卡片：DECODE · PREFILL · MTP 接受率 · MTP 草稿 · **KVMem 换块** · 提示 Tokens · **当前上下文** · 引擎运行；下方为本机资源占用（CPU / GPU 显存 / 系统内存 / GPU 状态）与对话 API 调用统计（每次生成追加一条：速度 · 用量），约 2 秒刷新 |
| 代理接口 | `/v1/models`、`/stats`、`/metrics`、`/props`、`/health` 原样代理；`/v1/chat/completions` 走 SSE 流式直通 |
| 本机信息 | `/api/sys` 用 ctypes + nvidia-smi 读 CPU/内存/显存；`/api/browse` 弹出系统原生文件/目录选择框（子进程 `pick.py`，Tk） |
| 自检 | `python verify.py`：校验 `index.html` 的 `FIELDS` 与 `launcher.py` 的 `FIELD_ORDER` 逐项一致、JS 引用的 DOM id / CSS class 均存在、`<script>` 配对且无未替换占位符、下拉取值都出现在 ninfer 服务端源码中；退出码 0 = 全部通过。**本仓库当前位置已实测通过**：59 项字段（基础 28 + 进阶 31）、61 个 DOM id、56 个 CSS 类、spec / kv_dtype / reasoning_effort / vision_residency 下拉取值全部与服务端源码一致 |

------------------------------------------------------------------------------------------------------------------------------------

256K 召回测试结果（2026-10-16，ninfer pin `1383a50e`）

**测试条件**：RTX 5060 Ti 16 GB（sm_120a）+ CUDA 13.4 + ninfer 后端；模型 `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp`（11.3 GiB，加载 5.4 s）；
KV 池 `36864 + 16384 = 53,248 token`（832 页 × 64）；ActKV 开启（`--kvmem-lrfu`，λ=0.50 / top-frac=0.90）；
`--max-context 262144`；**上下文峰值 258,382 / 262,144（98.6%）**；全程 12.2 分钟。

### 1. 三阶段召回（判定：精确子串匹配，值错一位即 MISS）

| 阶段 | 针数 | 结果 | 工具调用 |
|---|---|---|---|
| ① 同轮探针（每轮写入后立即问） | 33 | **33/33（100%）** | 33 |
| ② 最终召回（15 轮写完、历经全部驱逐后） | 33 | **33/33（100%）** | 33 |
| ② 最终召回（含重试并集） | 33 | **33/33（100%）** | —（**未触发任何重试**） |
| ③ 跨工具验证（改用 `send_alert` 通道） | 14 | **14/14（100%）** | 14 |
| **工具调用合计** | — | **80/80 成功** | **80** |

### 2. 逐轮写入 + 同轮探针（每轮 ≈16.6k token，共 15 轮）

| 轮 | prompt | 写入 prefill | 写入耗时 | 探针 prefill | 探针 decode | 本轮召回 | 工具 |
|---|---|---|---|---|---|---|---|
| 1 | 16,564 | **937.6** | 18.4 s | 234.1 | 68.6 | 3/3 | 3 |
| 2 | 16,192 | **880.9** | 18.6 s | 186.7 | 68.8 | 3/3 | 3 |
| 3 | 16,497 | **780.3** | 21.3 s | 142.9 | 73.9 | 3/3 | 3 |
| 4 | 16,587 | 378.7 | 44.0 s | 141.5 | 69.8 | 2/2 | 2 |
| 5 | 16,587 | 377.8 | 44.1 s | 124.9 | 70.3 | 2/2 | 2 |
| 6 | 16,587 | 377.1 | 44.2 s | 131.9 | 58.5 | 2/2 | 2 |
| 7 | 16,587 | 373.8 | 44.6 s | 138.0 | 52.5 | 2/2 | 2 |
| 8 | 16,587 | 376.6 | 44.3 s | 139.6 | 53.2 | 2/2 | 2 |
| 9 | 16,587 | 377.7 | 44.2 s | 139.9 | 62.7 | 2/2 | 2 |
| 10 | 16,588 | 375.7 | 44.5 s | 139.8 | 51.9 | 2/2 | 2 |
| 11 | 16,588 | 375.8 | 44.5 s | 130.4 | 57.0 | 2/2 | 2 |
| 12 | 16,588 | 375.1 | 44.6 s | 130.4 | 52.6 | 2/2 | 2 |
| 13 | 16,588 | 375.7 | 44.5 s | 137.0 | 56.7 | 2/2 | 2 |
| 14 | 16,588 | 376.4 | 44.4 s | 140.1 | 69.5 | 2/2 | 2 |
| 15 | 16,588 | 374.6 | 44.7 s | 135.0 | 57.7 | 2/2 | 2 |

### 3. 最终召回（258k 上下文，每批 11 针）

| 批次 | 针数 | 增量 prompt | prefill | **decode** | 耗时 | 工具 |
|---|---|---|---|---|---|---|
| 1 | 11 | **398** | 387.5 tok/s | **134.5** tok/s | 2.7 s | 11 |
| 2 | 11 | **398** | 385.1 tok/s | **108.4** tok/s | 3.0 s | 11 |
| 3 | 11 | **398** | 402.4 tok/s | **104.3** tok/s | 3.1 s | 11 |

`prompt 398` = 每批只新增 398 token，**其余 258k 全部缓存命中** —— 多轮会话的 KV 复用正常。

### 4. 跨工具验证（`send_alert`，逐针 14/14）

| 针 | 值 | 结果 | 耗时 | decode | 针 | 值 | 结果 | 耗时 | decode |
|---|---|---|---|---|---|---|---|---|---|
| N01 | N01-000000 | ✅ | 1.4 s | 58.3 | N08 | N08-120567 | ✅ | 1.4 s | 54.2 |
| N02 | N02-104729 | ✅ | 1.4 s | 53.6 | N09 | N09-225296 | ✅ | 1.4 s | 53.3 |
| N03 | N03-209458 | ✅ | 1.4 s | 54.3 | N10 | N10-023757 | ✅ | 1.5 s | 53.4 |
| N04 | N04-007919 | ✅ | 1.4 s | 53.9 | N11 | N11-128486 | ✅ | 1.4 s | 53.7 |
| N05 | N05-112648 | ✅ | 1.4 s | 54.7 | N12 | N12-031676 | ✅ | 1.4 s | 54.6 |
| N06 | N06-217377 | ✅ | 1.4 s | 54.7 | N13 | N13-136405 | ✅ | 1.4 s | 53.8 |
| N07 | N07-015838 | ✅ | 1.4 s | 54.4 | N14 | N14-039595 | ✅ | 1.4 s | 53.8 |

### 5. 服务端指标（`GET /stats`）

| 指标 | 值 |
|---|---|
| 上下文峰值（最后选择 frontier） | **258,382 / 262,144（98.6%）** |
| 请求 / 缓存来源 | 80 次（`root` 未命中 **2**、`private_endpoint` 命中 **78**） |
| **缓存复用** | 复用 prompt token **14,531,299**，重算仅 **252,910** → **98.3%** |
| KVMem 评估 token | 258,440 |
| 历史命中 / 未命中 | **78 / 2** |
| **页换入换出（history_swaps）** | **143 次** |
| 换出 / 换回字节 | spilled 17.2 GB / restored 188.8 GB |
| Host 归档占用 | **8.64 GiB / 12 GiB（72%）** |
| checkpoint 字节 / token | 461.9 MB / 258,440 |
| 统计表占用 | 273.1 MB（staging 2.8 MB） |
| 引擎忙时（host_work total） | 365.7 s（prefill device-wait 288.7 s） |
| 主 KV 迁移 | 0 字节（无 d2h/h2d 主 KV 搬运） |
| 失败与压力事件 | `context_cache_exhausted` 0、pressure 0、disk 0 |

### 6. 速度汇总

| 阶段 | min | 中位 | max |
|---|---|---|---|
| 写入 prefill tok/s | 373.8 | **376.6** | **937.6** |
| 同轮探针 prefill tok/s | 124.9 | 139.6 | 234.1 |
| 同轮探针 decode tok/s | 51.9 | 58.5 | 73.9 |
| 最终召回 decode tok/s | **104.3** | 108.4 | **134.5** |
| 跨工具 decode tok/s | 53.3 | 54.1 | 58.3 |

**双平台特征**：

```
轮 1–3（池未满，累计 <53k）  :  780 – 938 tok/s
轮 4–15（池已满，68k→258k）  :  374 – 379 tok/s   ← 12 轮几乎不变
```

从 68k 到 258k（3.8 倍上下文），池满后写入 prefill 只从 378.7 变到 374.6 —— **完全平坦**。掉速的分界线是“KV 池是否填满（53,248）”，不是上下文长度；代价是池满后约 375 tok/s，收益是 258k 与 68k 一样快。同轮探针 prefill 仅 125–234 tok/s 是因为探针只追加 ~118 token，**每 chunk 固定开销占主导**，不代表批量 prefill 能力。

### 7. 服务端日志摘录（同一轮测试的逐请求遥测）

| 观测 | 值 |
|---|---|
| **256K 下的缓存复用** | 每个请求 `cache` 命中率 **99.8–100.0%**（来源 `private endpoint`），`prompt` 每次只增长 ~118 token |
| **TTFT（256K）** | 工具调用轮 **0.83–1.8 s**；写作轮（每轮 +16.6k token、池未满时）18–45 s |
| **工具轮 decode（解码阶段）** | **111–188 tok/s**（例：`req#46 decode 188.5`、`req#48 185.4`、`req#50 159.4`） |
| MTP 接受率 | 纯 ACK 轮 **100%**（3/3）；工具轮混合推测 **42.7–75.4%**，ngram 另计（例 501/716、488/657） |
| 队列与总时长 | `queue` 0.3–0.6 s；单请求总时长 1.2–5.3 s |
| 主 KV 迁移 | 全程 0 字节（无 d2h/h2d 主 KV 搬运），`context_cache_exhausted` 0 |

> 服务端 `decode` 是**解码阶段**速率；§3 表里的是**请求级平均**，两者口径不同。

### 测试设计要点

1. **每轮写入必须 >16384 token** 才能持续压迫 KV 池（本测试 435 行 × 36.61 ≈ 16,588）；用 450 行 × 15 轮会到 260,948 token，第 2 批召回必然 `HTTP 400 context length exceeded`。
2. **每轮请求必须带相同的 `tools` 列表**，否则 chat 模板的 system 段变化会导致前缀失效、全量重算。
3. **工具调用必须回 `role=tool` + `tool_call_id`**，否则对话结构非法、模型停止调用工具。
4. 判定用**精确匹配**，同时报告“重试并集”（一位数字的笔误会被前者算作完全失败）。
