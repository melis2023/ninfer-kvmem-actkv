# KVMem · NINFER 图形化启动台

一个纯本地、零依赖的 Web 控制台，用来配置并管理
`infer-kvmem-server.exe`（ninfer 后端，Qwen3.8-27B KVMem 多后端框架）。

界面为仪表盘风格（默认**亮色**，可一键切换暗色），全中文，**单文件 HTML**
（内联 CSS + JS，无任何 JS 库、无构建步骤）。后端只用 **Python 3 标准库**，
仅监听 `127.0.0.1`。

---

## 快速开始

双击 **`start-launcher.bat`**（需要 PATH 中有 `python` 或 `py`）。

浏览器会自动打开 <http://127.0.0.1:18300>；若未自动打开，手动访问即可。

```bat
:: 或者手动运行
python launcher.py
:: 不自动开浏览器
python launcher.py --no-browser
:: 换一个启动器端口
python launcher.py --port 18301
```

---

## 界面布局

```
┌────────────────────────────────────────────────────────────────────────┐
│  04 实时指标（8 张卡片横排 · 占满整宽）                                   │
│  DECODE │ PREFILL │ MTP 接受率 │ 草稿 │ KVMem 换块 │ 提示 │ 上下文 │ 运行  │
│  本机资源占用： CPU 占用 │ GPU 显存 │ 系统内存 │ GPU 状态（带进度条）      │
│  对话 API 调用统计：单次对话列表（速度 · 用量）每次生成追加一条            │
├────────────────┬───────────────────────────────────────┬───────────────┤
│  01 服务控制    │  02  启动参数 │ 对话 │ 日志 ← 头部页签  │  03 预设管理   │
│  （330px）      │  同一张面板三页互斥，默认「启动参数」   │  （330px）     │
└────────────────┴───────────────────────────────────────┴───────────────┘
```

* **02 面板内置三页页签**：`启动参数` / `对话` / `日志` 是**同一张面板**（编号 02）里的三个页，
  页签在面板头部右侧（下划线样式，与参数表单里的分页 Tab 区分开）。默认停在「启动参数」；
  同一时间只显示一页，另两页只做 `display:none` —— 参数表单的填写、对话的会话与草稿、
  日志的环形缓冲与滚动位置都保留。选择记在 `localStorage` 的 `kvmem-launcher-view` 键下
  （刷新后仍停在原处）；切到「对话」/「日志」时各补一次滚到底（日志需勾选「跟随」）。
  序号：**01 服务控制 · 02（三页）· 03 预设管理 · 04 实时指标**，不再有 05 / 06。

* **顶部横排指标栏**：8 张指标卡片一行排开（DECODE / PREFILL / MTP 接受率 /
  MTP 草稿 / KVMem 换块 / 提示 Tokens / 当前上下文 / 引擎运行），占满整个宽度。
  一眼就能看到全部实时数据。
* **本机资源占用**（推理指标下方，同样每 2 秒刷新）：

  | 卡片 | 主数值 | 副标题 | 来源 |
  | --- | --- | --- | --- |
  | CPU 占用 | 占用率 % | 逻辑核数 | `GetSystemTimes` 两次采样差值 |
  | GPU 显存 | 显存占用 % | 已用 / 总量 MiB | `nvidia-smi --query-gpu=memory.used,memory.total` |
  | 系统内存 | 物理内存占用 % | 已用 / 总量 GiB | `GlobalMemoryStatusEx` |
  | GPU 状态 | GPU 利用率 % | 型号 · 温度 | `nvidia-smi --query-gpu=utilization.gpu,temperature.gpu` |

  每张卡片带一条**占用率进度条**，并按水位变色：<70% 正常色、≥70% 琥珀、
  ≥90% 红色。鼠标悬停 GPU 卡片可看完整信息。

* 响应式：推理指标 ≤1400px 折成 4 列、≤820px 2 列；资源卡片 ≤1000px 折成 2 列。
* **对话 API 调用统计**（在本机资源占用**下方**，**列表形式**，每完成一次生成追加一行）：
  按「单次对话」区分记录，最新在上，最多保留 10 条。记录**持久化**在浏览器
  `localStorage`（键 `kvmem-launcher-chatlog`）：**F5 刷新或重开浏览器后仍在**，
  且**只在点击「停止服务」成功时清空**（表示一轮服务已结束）。列：
  序号 / 时间 / 首字 TTFT / 输入 tokens（缓存 + prefill 分解）/ **Prefill 实际用量**
  （tok · 耗时 · tok/s）/ 输出 tokens / 生成速度 / 总耗时 / **实际上下文用量**。
  上下文用量按 `cache_n + prompt_n + predicted_n` 计算，并对照
  `/stats.capacity.max_context` 给出百分比。数据取自上游 SSE 最后一帧的
  `timings`（请求带 `timings_per_token: true`），**只存速度与用量，不存对话内容**。

| 面板 | 名称 | 说明 |
| --- | --- | --- |
| 01 | **服务控制** | 启动 / 停止受管进程；显示进程状态、PID、启动时间、退出码、上游健康度、上游地址；底部实时显示**将要执行的完整命令行**，以及可「浏览…」自选的 worker **EXE 路径**与**工作目录** |
| 02 | **启动参数 / 对话 / 日志** | **三页共存于同一张面板**（头部页签切换，`localStorage` 记忆）：① 启动参数——59 项参数表单，**用 10 个 Tab 分页**（模型与端口 / KVMem 内存 / actkv / 精度与采样 / MTP 投机解码 / 视觉输入 / 推理与附加 / 进阶·KVMem 调优 / 进阶·精度与内核 / 进阶·长上下文），支持**同步运行中参数**与**还原主用预设**；② 对话——多会话聊天页（左侧会话列表 + 右侧消息流，Markdown 渲染、代码块复制、逐条耗时与 AI 思考折叠；详见下方「对话页」）；③ 日志——受管进程 stdout/stderr 环形缓冲（8000 行），支持关键词过滤与自动跟随 |
| 03 | **预设管理** | 保存为当前预设、另存为新预设、载入、**改名**（按钮或双击标题）、设为主用、删除；卡片显示关键参数标签。**蓝框 = 当前选中**（点击载入的那张），**「● 主用」徽标 = 主用预设**，两者独立 |
| 04 | **实时指标** | 位于页面顶部、8 张卡片横排；每 2 秒刷新，上游不可达时显示「—」并注明原因。内含**「对话 API 调用统计」**列表：按单次对话记录速度/用量（TTFT、prefill 实际用量、生成速度、总耗时、实际上下文用量） |

顶部状态灯：<span>●</span> 绿=受管进程运行中，琥珀=启动中，红=已退出，
青=**检测到外部运行的服务**（不受本启动器管理），灰=未运行。

### 对话页（多会话 · 可当聊天页日常使用）

`02` 面板的**对话**页已从「单轮测试框」升级为**完整聊天页**，布局为左侧会话列表 + 右侧聊天区。

* **多会话**：`＋ 新建对话` 建新会话；点击列表项切换；双击/直接编辑标题栏改名；
  列表项右侧 `×` 删除（有消息时二次确认）。会话按最近使用时间排序。
  全部数据保存在浏览器 `localStorage` 的 `kvmem-launcher-chats` 键下，**刷新或重开浏览器后仍在**。
* **多轮上下文**：发送时会把该会话最近的若干轮消息（由「上下文轮数」控制，默认 12 轮 ≈ 24 条）
  连同可选的 **system 提示词**一起发给上游 `/v1/chat/completions`。历史会被自动裁剪，
  保证首条一定是 user 消息，避免长对话撑爆上下文。
* **流式渲染**：SSE 逐字追加，带闪烁光标；输入框下方「实时指标条」显示
  首字延迟 / 流式进度 / 总耗时 / tok/s / MTP 接受率。**每条助手消息下方**另有可折叠的
  「耗时明细」表（缓存命中、Prefill、生成、草稿、接受率等全部 `timings` 字段）。
* **思考内容**：若上游在 SSE 分片里返回 `delta.reasoning_content`（思考/推理内容，
  通常先于正文到达），会在助手气泡**顶部**以可折叠的「思考内容 · N 字」区块展示，
  **默认展开、内容完整铺开（不截断、不做内部滚动）**，点击标题可收起；
  流式期间实时增长、生成结束后随会话一起保存（`localStorage`）。正文与思考分隔渲染，
  思考文本按纯文本转义输出，不参与 Markdown/HTML 解析。
* **消息操作**（鼠标悬停在消息上出现）：复制、`编辑重发`（用户消息，载入输入框并截断其后历史）、
  `重新生成`（助手消息）、`删除`。
* **停止生成**：生成中发送按钮变为 `停止`，点击即中止（`AbortController`），**已生成的部分会保留**。
* **顶部工具条**：可编辑的会话标题、当前模型名、`导出`（当前会话导出为 JSON）、`清空`（清空消息但保留会话）。
* **对话设置**（折叠区，**按会话独立保存**）：system 提示词、`max_tokens`（**默认留空 = 不限制**，见下）、`temperature`、`top_p`、上下文轮数。

  > **输出长度怎么控制**：默认**两边都不限制** —— 对话页不发送 `max_tokens`，
  > 启动参数也不设 `--default-max-tokens`，此时服务端用 `max_context` 作为上限（实际受上下文占用约束）。
  > 想限制单次输出，用 **KVMem 内存页签的「单次最大输出」**（服务端 `--default-max-tokens`），
  > 或在对话设置里给该会话填一个 `max_tokens`（仅覆盖本次请求）。上下文大小始终由
  > `--max-context` 决定，**与输出上限无关**。
  > 若输出被上限截断，上游会以 `finish_reason=length` 结束，消息下方会出现
  > 「⚠ 已达输出上限，内容被截断」提示，并弹出同样的提示条。
* **Markdown 渲染**：标题、**粗体**、*斜体*、~~删除线~~、`行内代码`、围栏代码块（右上角带「复制」）、
  有序/无序列表、引用、分隔线、链接。渲染**先整体 HTML 转义、再套用自己生成的标签**，
  链接仅允许 `http(s)`，因此模型输出无法向页面注入 HTML/脚本。
* **空状态**：无消息时展示 4 条示例提示，点击即填入输入框。
* **滚动**：新消息自动滚到最新；若你向上翻阅，右下角会出现 `↓ 回到最新`。

> ⚠️ 会话数据存在**浏览器本地**（`localStorage`），不写入 `config.json`，也不上传到 launcher。
> 换浏览器/清站点数据会丢失；需要留存请用「导出」。

### 启动参数 Tab

59 个字段按语义分为 10 页，点页签切换（同一时间只显示一页）：

| Tab | 字段数 | 内容 |
| --- | --- | --- |
| 模型与端口 | 5 | 模型文件（带「浏览…」）、对外模型 ID、服务端口、最大上下文、**GPU 设备**（`--device`） |
| KVMem 内存 | 7 | **启用 KVMem 总开关**、近期 token、分块大小、预算、生成预留、主机预算、**单次最大输出**（留空＝不限） |
| actkv | 4 | LRU 淘汰、LRFU 衰减 λ、每轮保留比例、§8.1 语义保护 |
| 精度与采样 | 5 | KV 量化类型、温度、Top-P、Top-K、Min-P |
| MTP 投机解码 | 4 | 投机方式、MTP 草稿数、N-gram 草稿数、自适应 MTP |
| 视觉输入 | 3 | 视觉开关、驻留位置、最大合并 |
| 推理与附加 | 3 | 推理强度、思考预算、附加参数 |
| 进阶·KVMem 调优 | 15 | 并发、锚定 token、会话数、自适应预算、SSD 分层等 |
| 进阶·精度与内核 | 11 | cuBLAS、量化、CUDA Graphs、Prefill 内核 |
| 进阶·长上下文 | 3 | YaRN 与线性插值（RoPE） |

* 页签选择记在 `localStorage`，刷新后回到上次那一页。
* **隐藏页签里的字段仍然存在于 DOM**，因此「保存预设」「同步参数」
  「命令行预览」始终读取全部 59 项，不会因为切页而丢参数。

> **KVMem 总开关**：「KVMem 内存」页签的第一项是**启用 KVMem**（**没有对应命令行标志**，
> 纯开关）。勾选时正常拼接 `--kvmem-*`；**取消勾选后启动命令会丢弃全部 `--kvmem-*`**
> （含 `--kvmem-budget` / `--kvmem-gen-reserve`），退回 **ninfer 原版启动方式**。
> 之所以必须整组丢弃：服务端在没有 `--kvmem-budget` 时会**拒绝任何 KVMem 辅助参数**
> （`serve_options.cpp`：*"KVMem auxiliary options require --kvmem-budget"*），
> 只去掉 budget 而留着其余参数会直接启动失败。
> 关闭时其余 KVMem 输入框会**置灰**（值仍保留，便于再次开启），而 `--max-context`、
> `--prefill-chunk`、`--kv-dtype`、`--vision*` 等**非 KVMem 参数不受影响**。
> 旧配置若没有这个键，一律按**开启**处理，避免升级后静默改变行为。

### 进阶参数（31 项，共 59 项字段）

进阶字段取自 `serve_options.cpp` 的 `ADVANCED_ORDER`，全部与当前 KVMem 配置兼容。
每个字段的取值区间都来自服务端校验代码，界面会显示在标志旁。

> 其中 1 项 ActKV 保留策略参数（§8.1 语义保护）在**展示上**并入
> `actkv` 页签（见下表），因此页签字段数之和仍是 58。

**actkv（4）**：ActKV 淘汰与保留策略集中在这一个页签（`--kvmem-lrfu`
属基础参数，展示上并入此页）

| 参数 | 标志 | 取值/默认 | 说明 |
| --- | --- | --- | --- |
| KVMem LRU 淘汰 | `--kvmem-lrfu` | 开关 | LRU/LRFU 淘汰策略总开关 |
| LRFU 衰减 λ | `--kvmem-lrfu-lambda` | `(0,1)` 默认 0.50 | |
| 每轮保留比例 | `--kvmem-lrfu-top-frac` | `(0,1]` 默认 0.90 | |
| 保护语义块（§8.1） | `--kvmem-protect-semantics` | 开关，默认关 | 钉住前导 system/developer 提示词与当前 user 任务，窗口永不压缩它们 |

> **§5.1 AAV 已从上游移除**：`--kvmem-aav-protect`、`--kvmem-aav-lambda-intermittent`、
> `--kvmem-aav-min-hits` 三个标志在本服务端会直接报错（见 `serve_options.cpp`：论文 §5.1 的
> AAV 只是**诊断**，并不定义按类别区分的衰减；断续块同样只靠唯一的全局
> `--kvmem-lrfu-lambda` 存活）。因此本启动器**不再暴露**这些字段。分类信息仍可
> 通过环境变量 `KVMEM_AAV_TRACE=1` 采集。

**进阶·KVMem 调优（15）**

| 参数 | 标志 | 取值/默认 | 说明 |
| --- | --- | --- | --- |
| 并发解码请求数 | `--max-concurrency` | `[1,8]` 默认 1 | 同时解码的请求数 |
| KVMem 锚定 token | `--kvmem-sink-tokens` | 64 对齐，0=1 页 | 固定保留的前缀 |
| 保留历史会话数 | `--kvmem-sessions` | `1..16` 默认 4 | 保留的 Host 历史 |
| 按工具调用评分 | `--kvmem-action-region` | 开关 | ActKV 5.1 |
| 双查询并集评分 | `--kvmem-query-union` | 开关 | 修掉上一项的窗口丢失 |
| 自适应预算起点 | `--kvmem-conf-budget` | 64 对齐，0=关闭 | 及 `scale-up/cooldown/pool/stride/bottom` 共 6 项 |
| SSD 分层配额 | `--kvmem-disk-mib` | MiB | 需配合 `--kvmem-disk-path`（放在附加参数） |
| 共享 KV 容量 | `--kv-capacity` | `auto` 或 token 数 | 须 ≥ `--max-context` |
| 默认输出上限 | `--default-max-tokens` | ≥1 | 请求未指定时的上限 |
| 关闭前缀缓存 | `--no-prefix-reuse` | 开关 | |

**进阶·精度与内核（11）**：`--fast-prefill-kernel`、`--prefill-cublas`、
`--mlp-a8-decode`、`--no-prefill-a8`、`--lm-head-q4/q6`、`--embedding-q4/q6`、
`--gdn-state-fp16`、`--no-cuda-graph`、`--cuda-graph-allowance-mib`

**进阶·长上下文（3）**：`--rope-yarn`、`--rope-yarn-factor [1,4]`、
`--rope-scaling-factor [1,32]`

**没有加进来的参数**（源码明确与 KVMem 冲突或不适用）：

| 标志组 | 原因 |
| --- | --- |
| `--model-suspend` `--suspend-*` `--no-auto-resume` | `serve_options.cpp:1655` — KVMem 不支持模型挂起 |
| `--graft` `--default-graft` | 同上，不支持 prompt grafts |
| `--models-dir` `--models-preset` `--models-max` `--no-models-autoload` | router 模式与位置参数模型互斥 |
| `--lookup-ngram` | `:1873` — KVMem 禁用此旧标志 |
| `--expert-residency` `--expert-cache-mib` `--ngram-table` 等 | Qwen3.8-Flash-Next 专用，本模型不适用 |
| `--devices` `--stage-layers` | 多卡流水线，本机单卡 |

### ⚠️ 两个隐性规则

1. **`--ngram-draft-tokens` > 15 且 `--max-concurrency` > 1 时会被压到 15**
   （`speculative_limits.h:11`）。不报错，但会静默生效。界面在
   `ngram_draft_tokens` 的区间提示里标了 `[0,63]`，若同时把并发调到 >1，
   实际生效值会是 15。
2. **`--kvmem-disk-mib` 必须配合 `--kvmem-disk-path`**，
   否则 SSD 分层不生效（`KVMem SSD requires both --kvmem-disk-path and --kvmem-disk-mib`）。
   路径字段放在「附加参数」里手填。

### 路径自选按钮（模型 / EXE / 工作目录）

界面上有 **3 个「浏览…」按钮**，直接弹系统原生选择框，省得手抄长路径：

| 位置 | 字段 | 选择框类型 |
| --- | --- | --- |
| 面板 02 · 模型与端口 | 模型文件 | 文件，过滤 `*.ninfer` |
| 面板 01 · 服务控制 | worker **EXE** | 文件，过滤 `*.exe` |
| 面板 01 · 服务控制 | **工作目录**（`bin_dir`） | 目录 |

为什么对话框要由**后端**弹（`POST /api/browse` → 子进程 `pick.py`）：
浏览器出于安全限制拿不到本地真实路径（`<input type=file>` 只给文件名和假路径），
而启动器与浏览器本来就在同一台机器上，所以由 `launcher.py` 起一个 Tk 子进程弹框，
选中的完整路径经 stdout 回传，前端再填进输入框。

行为约定：

* **起始目录**取当前字段的值：给了文件就定位到它所在目录，方便"就近挑一个"。
* **取消不是错误**：后端返回 `{ok:false,cancelled:true}`，界面静默，不弹红条。
* 路径里的空格和中文都安全 —— 子进程**按参数列表**启动（不经过 shell），
  且父子进程统一用 UTF-8 传路径。
* 对话框弹在**启动器所在的机器**上；若被浏览器窗口挡住，用 Alt+Tab 找一下
  （`pick.py` 已尽量设为置顶）。
* 模型路径属于**预设**（改完点「保存为当前预设」，或直接点启动——启动时会自动存）；
  **EXE 与工作目录属于启动器配置**，改完立刻写回 `config.json`（不会被动过预设）。
* 想单独排查选择框本身：可手工跑
  `python pick.py --mode file --initial "I:\model" --types "ninfer 模型 (*.ninfer)|*.ninfer"`，
  退出码 0=选中 / 1=取消 / 3=tkinter 不可用。

> 需要 `tcl/tk`（Python 官方 Windows 安装包默认带）。若缺失，`pick.py` 以退出码 3
> 返回，界面提示「选择失败：tkinter 不可用: …」，此时手填路径仍然可用。

---

## 亮色 / 暗色主题

**默认是亮色（白色）界面**，右上角有主题切换按钮：

| 方式 | 操作 |
| --- | --- |
| 鼠标 | 点右上角 `☀ 亮色` / `☾ 暗色` 按钮 |
| 键盘 | 按 **T** 键（在输入框内打字时不会误触发） |

* 默认固定为亮色，**不跟随系统主题**，保证"打开就是白色"。
* 手动切换后会记在 `localStorage`，**刷新后保持**。
* 两套主题共用同一组 CSS 变量（`:root` 是亮色，`[data-theme="dark"]` 覆盖为暗色），
  即使 JS 被禁用也仍是白色界面；切换时面板、按钮、输入框、卡片会同步过渡。

---

## 中文输入（IME）

对话输入框完整支持中文/日文输入法：

* 输入法**组合期间按 Enter 确认候选词，不会误触发送**，也不会被
  `preventDefault()` 吞掉导致候选词上不去屏。
* 发送用 **Enter**，换行用 **Shift+Enter**。

不同浏览器/输入法的按键序列并不一致，代码里逐条覆盖：

| 场景 | 特征 | 处理 |
| --- | --- | --- |
| Chrome / Edge（微软拼音） | `keydown` 且 `isComposing=true`、`keyCode=229` | 拦截 |
| Firefox | `keydown` 且 `isComposing=true` | 拦截 |
| Safari / 部分输入法 | `keyCode=229` 但 `isComposing=false` | 拦截 |
| 部分实现 | `key === "Process"` | 拦截 |
| 少数实现 | **`compositionend` 先于 `keydown(Enter)` 到达** | 80ms 时间窗兜底 |

最后一条是最隐蔽的：同一次上屏按键，如果 `compositionend` 先触发，组合标志已被
清除，紧随其后的 Enter 就会被误判为"发送"。因此额外记录 `compositionend` 时间戳，
80ms 内的 Enter 一律交还给输入法。

> 另外，`/` 与 `/index.html` 响应带 `Cache-Control: no-store`。
> 早期版本漏了这个头，导致改了前端后浏览器仍可能用旧页面，
> 表现为"改了没生效"。

---

## 参数与命令行构造

表单字段与 `launcher.py` 中的 `FIELD_ORDER` + `ADVANCED_ORDER` **逐字一致**，
命令行构造规则也与随附的 `start-kvmem-ninfer.bat` 语义相同：

1. 第一个 token 是 exe 路径；
2. 若「模型文件」非空，则作为**位置参数**紧跟其后；
3. 其余字段按 `FIELD_ORDER`（基础）→ `ADVANCED_ORDER`（进阶）顺序输出 `--flag value`；
4. `bool` 字段勾选才输出该 flag（不输出值）；
5. 文本/数值字段为空串、`"0"` 或未提供时**整项省略**；
6. 「附加参数」按空白拆分后**原样追加**到末尾。

构建期会用脚本把前端 `buildCmd()` 与 Python `build_command()` 做
**交叉校验**（18 个用例逐 token 比对），确保界面预览与实际启动的命令完全一致。

### 参数一览（25 项）

| 参数 | 标志 | 说明 |
| --- | --- | --- |
| 模型文件 | *(位置参数)* | `.ninfer` 模型路径 |
| 对外模型 ID | `--model-id` | OpenAI 接口返回的模型名 |
| 服务端口 | `--port` | 上游监听端口，默认 18200（**下拉/区间 [1,65535]**） |
| 最大上下文 | `--max-context` | 例如 262144 |
| GPU 设备 | `--device` | CUDA 设备序号（0 = 第一块卡，省略即服务端默认 0）。下拉候选来自本机 `nvidia-smi` 采样；拿不到显卡列表时退化为数字输入。多卡流水线请用 `--devices` + `--stage-layers`，写进「附加参数」 |
| KVMem 近期 token | `--kvmem-recent-tokens` | 例如 1024（≥64，64 对齐） |
| Prefill 分块大小 | `--prefill-chunk` | 例如 256（**须为 128 的正整数倍**） |
| KVMem 预算 token | `--kvmem-budget` | 例如 36864（64 对齐） |
| 生成预留 token | `--kvmem-gen-reserve` | 例如 16384（64 对齐） |
| 主机 KV 预算 MiB | `--kvmem-host-mib` | 例如 12288 |
| KVMem LRU 淘汰 | `--kvmem-lrfu` | 开关 |
| KV 量化类型 | `--kv-dtype` | **下拉：9 种服务端支持的格式** |
| 默认温度 | `--temperature` | **[0,2]** |
| 默认 Top-P | `--top-p` | **[0,1]** |
| 默认 Top-K | `--top-k` | **[0,20]** |
| 默认 Min-P | `--min-p` | **[0,1]** |
| 投机解码方式 | `--spec` | **下拉：mtp / dflash / dflash2** |
| MTP 草稿数 | `--draft-tokens` | **[1,15]** |
| N-gram 草稿数 | `--ngram-draft-tokens` | **[0,63]** |
| 自适应 MTP | `--adaptive-mtp` | 开关（仅 `--spec mtp` 有效） |
| 视觉输入 | `--vision` | 开关 |
| 视觉驻留位置 | `--vision-residency` | **下拉：resident / overlay / cpu** |
| 视觉最大合并 | `--vision-max-merged` | **[64,16384]**，默认 **256**（`--vision-residency cpu` 时）/ 16384 |
| 默认推理强度 | `--default-reasoning-effort` | **下拉：模板实际只支持 3 档**（见下） |
| 默认思考预算 | `--default-thinking-budget` | 例如 53248 |
| 附加参数 | *(原样追加)* | 其他未列出的标志 |

### 下拉菜单与取值约束

以下 4 个参数改用**下拉菜单**，取值**全部取自 ninfer 服务端源码的枚举解析函数**
（不是凭经验写的），因此不可能选出服务端会拒绝的值：

| 参数 | 下拉取值 | 来源（源码） |
| --- | --- | --- |
| `--spec` | `mtp` `dflash` `dflash2` | `product/speculative_options.h` → `parse_speculative_backend()` |
| `--kv-dtype` | `bf16` `int8` `fp8` `rk8v4` `rk4v4` `rk4v4-e8` `rk2v4-e8` `nvfp4` `k8v4` | `serve/serve_options.cpp` → `parse_kv_dtype()` |
| `--vision-residency` | `resident` `overlay` `cpu` | `serve/serve_options.cpp`（`--vision-residency` 分支） |
| `--default-reasoning-effort` | 见下方「推理强度」小节 | `serve/request.h` + `tools/chat_templates/qwen3_8.jinja` |

* 每个下拉都带一个**空选项**「（省略）」，选中即不传该参数、沿用服务端默认值。
* 下拉项显示中文说明（例如 `rk8v4 · 旋转 INT8 键 + int4 值`），鼠标悬停有完整提示。
* 数值字段按服务端校验区间设置了 `min`/`max`/`step`，并在标志旁显示区间提示
  （如 `--port [1,65535]`、`--prefill-chunk 128 的正整数倍`）。
* 「同步运行中参数」若读到下拉列表外的值（服务端版本差异），会**临时补一个选项**
  并标注「来自运行中的服务」，不会静默丢参数。

#### 推理强度：协议层 7 档 ≠ 模板层 3 档

这是一个容易踩的坑，**两个层次必须分开看**：

| 层次 | 接受的值 | 源码 |
| --- | --- | --- |
| CLI / HTTP 协议层 | `none` `minimal` `low` `medium` `high` `xhigh` `max`（7 档） | `serve/request.h` → `parse_requested_reasoning_effort()` |
| Qwen3.8-27B chat template | **只有 `xhigh` `medium` `low`** | `tools/chat_templates/qwen3_8.jinja`（`not in ('xhigh','medium','low')` 即报错） |

模板不认的值**不会报错**，而是由 `accepted_effort()` **就近映射**（平局向上取），
`translate.cpp` 的注释也确认了这一点：

| 你选择 | 实际生效 |
| --- | --- |
| `xhigh` / `medium` / `low` | 原样生效（模板原生支持） |
| `high` | → `xhigh` |
| `max` | → `xhigh` |
| `minimal` | → `low` |
| `none` | 关闭思考（不发思考提示） |

所以下拉里**前 4 项按模板原生支持的档位排在前面**，其余项都在标签里显式标注
`→ 映射为 xxx（模板不支持）`，避免让人误以为 `high` 是一个独立档位。

> `verify.py` 会**重新解析 C++ 源码与 chat template**，逐值比对下拉选项，
> 并检查模板不支持的值是否已标注映射关系，确保选项始终是服务端真实支持的。

> 说明：数值字段在界面里是文本框，提交的是**字符串**。
> 因此 `1.0` 会写成 `--temperature 1.0`；而预设里的数字 `1.0` 在服务端路径
> 会变成 `1`。引擎默认值本就是 `temperature=1.0F`、`min_p=0.0F`，
> 且 `min-p 0` 与省略等价，故两种写法**语义完全相同**。

---

## 预设（config.json）

预设与配置保存在 `launcher.py` 同目录的 **`config.json`**，采用
「临时文件 + 原子替换」写入，避免中断导致配置损坏。

内置 3 个预设（来自 `start-kvmem-ninfer.bat` 的对照配置）：

1. **MTP + ActKV（bat 默认 · 当前运行）** — `--spec mtp`、`--kv-dtype rk8v4`
2. **DFlash2 + nvfp4 权量** — `--spec dflash2`、`--kv-dtype nvfp4`
3. **无 ActKV 基线（对照）** — 不启用 `--kvmem-lrfu`

三个预设都带 `kvmem_protect_semantics: true`（与 `start-kvmem-ninfer.bat` 的
`--kvmem-protect-semantics` 对齐，保证对比时只有 LRU 这一个变量在变）；
§5.1 AAV 相关标志已被上游移除，预设里不再包含。

在面板 02 改完参数后，点「保存为当前预设」写入**当前选中（蓝框）的预设**，或「另存为新预设」
新建一份；面板 03 的卡片可随时「载入 / 改名 / 设为主用 / 删除」（改名也可双击卡片标题）。
改名只改名称、保留该预设的参数，也不影响「主用」与当前选中项。

> **蓝框 vs 主用**：卡片上的**蓝色边框表示「当前选中/已载入」**（点哪张就跟随哪张，与表单内容对应）；
> **「● 主用」徽标才是主用预设**（启动服务时默认采用、`--kvmem-*` 只由它决定）。两者相互独立：
> 选中一张不等于设为「主用」，「设为主用」也不会抢走你的选中态。

**同步运行中参数**：从上游 `/v1/models` 的 `status.args` 读取**实际生效**的
启动参数并回填表单（共识别 25 个标志中的对应项），便于「照着正在跑的服务改」。

---

## 指标来源

| 指标 | 来源 |
| --- | --- |
| DECODE / PREFILL 速度 | `/metrics` → `llamacpp:predicted_tokens_seconds` / `llamacpp:prompt_tokens_seconds`（执行时间平均 tok/s，与 llama.cpp eval rate 同口径）；无此指标时回退 `/stats` → `throughput_tokens_per_second.{decode,prefill}`（总运行时间摊销值，含空闲期，偏低） |
| MTP 接受率 | `/metrics` → `ninfer:draft_accepted_tokens_total / ninfer:draft_tokens_total` |
| KVMem 换块 | 主数值＝换块次数 `/stats` → `kvmem.history_swaps`（KV 超过 `budget + reserve` 分区后，每轮压缩把页换出到主机、把选中的页换回显存各记一次）；副标题＝换出 `kvmem.history_spilled_bytes` / 换入 `kvmem.history_restored_bytes`（GiB）与**宿主路径延迟** `host_work.detail_subset_seconds.context_progress` ÷ `detail_invocations.context_progress`（ms/轮，该区域含 KVMem 选页与换出/换入）。**单次换块的纯 GPU 拷贝毫秒不在 `/stats` 里**，只在 `KVMEM_ADVANCE_TRACE=1` 运行时打印；次数为 0 表示 KV 还没超过 B+R 分区 |
| 提示 Tokens | `/stats` → `tokens.computed_prefill` |
| 当前上下文 | 主数值＝当前上下文 token，紧随其后的灰字＝`/最大上下文`：优先取「对话 API 调用统计」里**最近一条**的 `ctx_total`（`cache_n + prompt_n + predicted_n`）；刚打开页面还没有请求记录时，回退 `/stats` → `kvmem.checkpoint_tokens`（服务端当前保留的上下文）。分母为 `/stats` → `capacity.max_context`，副标题显示占用百分比 |
| 引擎运行时长 | `/stats` → `uptime_seconds` |
| 解码轮次 | `/metrics` → `llamacpp:n_decode_total` |
| Prefill 耗时 | `/metrics` → `llamacpp:prompt_seconds_total` |
| 缓存命中 | `/metrics` → `ninfer:reused_prompt_tokens_total` |

指标每 2 秒刷新一次，**仅在上游健康（`/health` 返回 `status:"ok"`）时拉取**；
上游不可达时卡片会显示「—」并注明原因，不会把旧值当作实时数据。

对话页的耗时明细来自 SSE 分片里的 `timings` 字段（请求时携带
`timings_per_token: true`），键为：`cache_n`、`prompt_n`、`prompt_ms`、
`prompt_per_token_ms`、`prompt_per_second`、`predicted_n`、`predicted_ms`、
`predicted_per_token_ms`、`predicted_per_second`、`draft_n`、`draft_n_accepted`。
每条助手消息各自保存自己的 `timings`，在气泡下方折叠展示（见「对话页」）。

---

## 后端接口

启动器本身监听 `127.0.0.1:18300`，提供以下接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 控制台页面 |
| GET | `/api/state` | 进程与上游状态快照 |
| GET | `/api/config` | 当前配置（含预设） |
| POST | `/api/config` | 部分键合并保存 |
| POST | `/api/browse` | 弹出系统文件 / 目录选择框（子进程 `pick.py`），返回所选**完整路径**；用户取消返回 `{ok:false,cancelled:true}`（HTTP 200） |
| GET | `/api/server/log?after=N` | 增量日志 `{lines,next}` |
| GET | `/api/sys` | 本机 CPU / 内存 / GPU 占用（后台每 2 秒采样，读取缓存，不阻塞） |
| POST | `/api/server/start` | 启动受管进程 |
| POST | `/api/server/stop` | 停止受管进程（`taskkill /F /T`） |
| GET | `/api/upstream/<path>` | 原样代理上游（`/stats`、`/metrics`、`/v1/models` …） |
| POST | `/api/upstream/chat` | SSE 直通 `/v1/chat/completions`（对话页使用；会话历史与设置在**浏览器端**维护，服务端无状态） |

---

## 外部服务守卫（重要）

启动前会先探测上游端口：

* 若该端口**已有服务在运行**（例如手动用 bat 启动、或别的程序占用），
  启动会被拒绝并提示「端口 xxx 已检测到运行中的服务（不受本启动器管理）」，
  同时界面显示青色状态灯与「外部服务」。
  这样做是为了避免**重复拉起 27B 模型**导致显存溢出。
* 若要由启动器接管，请先停掉外部服务，或把「服务端口」改成别的值。
* 「停止服务」只会结束**本启动器自己启动的**进程，不会去杀外部服务。
* 同理，**关闭启动器窗口也不会影响外部服务** —— 自动清理只作用于自己启动的进程。

---

## 关闭启动器时自动停止模型服务（重要）

**关闭 `start-launcher.bat` 窗口（或 Ctrl+C、或结束 python 进程）时，
由本启动器启动的模型进程会被一并结束，不会留下孤儿进程占着显存。**

实现分两层，互为补充：

| 层 | 机制 | 覆盖场景 |
| --- | --- | --- |
| **内核兜底** | Windows **作业对象（Job Object）** + `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` | 任何情况，**包括进程被强杀（`taskkill /F`）、任务管理器结束进程、断电前的强制终止** |
| **优雅清理** | `SetConsoleCtrlHandler` 捕获 `CTRL_CLOSE_EVENT` / 注销 / 关机，退出前主动停止模型 | 点击窗口 × 正常关闭、注销、关机 |

作业对象的工作方式：模型进程在 `start()` 时被加入一个作业，作业句柄由启动器持有。
**句柄随启动器进程结束而关闭，内核随即强制结束作业内所有进程** —— 这是操作系统级
保证，不依赖 Python 代码跑到清理逻辑，所以对强杀同样有效。

安全性：作业对象**只包含本启动器自己启动的进程**。如果 27B 服务是你手动用
`start-kvmem-ninfer.bat` 启动的（不受本启动器管理），关闭启动器**不会**影响它。

---

## 文件说明

```
kvmem-launcher/
├── start-launcher.bat   启动入口（双击即可）
├── launcher.py          Python 标准库后端（HTTP + 进程管理 + 配置持久化）
├── pick.py              原生文件/目录选择框（Tk 子进程，供 /api/browse 调用）
├── verify.py            一致性自检（前端 FIELDS ↔ launcher.py FIELD_ORDER）
├── README.md            本文件
├── config.json          运行后自动生成：端口、预设、路径
└── web/
    ├── index.html       单文件前端（内联 CSS + JS）
    └── index.html.bak   改造前的备份（可删；回滚时覆盖回 index.html 即可）
```

### 一致性自检

```bat
python verify.py
```

检查前端 `web/index.html` 与 `launcher.py` 是否仍然一致：

1. `FIELDS` 与 `FIELD_ORDER` 逐项相同（顺序 / 键名 / 标志 / 类型）——
   这是面板 01 命令行预览正确的前提；
2. JS 引用的每个 DOM id 都存在于页面；
3. JS 使用的每个 CSS class 都已定义；
4. **下拉菜单的每个取值都必须在 ninfer 服务端源码中出现过**
   （直接解析 `parse_kv_dtype()` 等函数，保证「选项都是服务端真实支持的」）；
5. **推理强度的模板层校验**：解析 `qwen3_8.jinja` 拿到模板真正支持的档位，
   确认下拉里模板不支持的值都已在标签中标注「映射为 xxx」——
   防止把会被静默重映射的值伪装成独立档位（这是一个曾经真实存在的缺陷）；
6. 结构完整（`<script>` 配对、无未替换占位符）。

改动任一侧的参数字段或下拉选项后，请运行一次 `verify.py`。
第 4、5 项需要能访问 ninfer 源码目录；找不到时会自动跳过并提示。

---

## 常见问题

**Q：改了前端代码，刷新页面却没变化？**
A：正常情况下不会 —— `/` 已带 `Cache-Control: no-store`。若仍异常，
用 `Ctrl+F5` 强制刷新，或检查是否有代理缓存。

**Q：推理强度里没有独立的 `high` 档？**
A：`high` 会被 Qwen3.8 模板映射为 `xhigh`，下拉里已标注。这是模型的
chat template 决定的，不是启动器的限制，详见上文「推理强度」小节。

**Q：页面打开是空白 / 一直显示「尚未连接启动器后端」？**
A：说明 `launcher.py` 没在运行，或端口不是 18300。先运行
`start-launcher.bat`，并确认 18300 未被占用。

**Q：顶部显示「外部服务」且「启动服务」是灰的？**
A：目标端口已有服务在跑，这是**保护机制**（防止重复加载 27B）。
先停止外部服务，或在面板 02 改端口。

**Q：指标全是「—」？**
A：上游不可达。看面板 01 的「上游健康」和 `02` 面板「日志」页。

**Q：改了参数但启动的命令没变？**
A：面板 01 底部的命令行是实时预览。若显示的是**受管进程运行中**的
实际命令行，说明进程正在运行——参数改动要等下次启动才生效。

**Q：想手动指定 Python 解释器？**
A：`start-launcher.bat` 会依次尝试 `python`、`py`；也可直接
`C:\path\to\python.exe launcher.py`。

**Q：对话记录存在哪里？换台电脑还在吗？**
A：存在**当前浏览器**的 `localStorage`（键 `kvmem-launcher-chats`），
launcher 后端不保存、也不上传。换浏览器或清站点数据会丢失，
重要对话请用对话页右上角「导出」保存为 JSON。

**Q：对话发出去没反应 / 报「上游服务不可用」？**
A：先确认面板 01 的上游健康为绿（服务已启动）。对话请求走
`POST /api/upstream/chat` 直通 `/v1/chat/completions`，上游没起来就发不出去。

**Q：多轮对话会不会把上下文撑爆？**
A：不会。每次只发送该会话**最近 N 轮**（对话设置里的「上下文轮数」，默认 12 轮 ≈ 24 条），
并保证首条一定是 user 消息。要保留更长记忆可调大该值，但要留意上游的
`--max-context` 预算。

**Q：生成到一半能停吗？**
A：能。生成中发送按钮变成 `停止`，点击即中止（`AbortController`），
**已经生成的内容会保留**并可继续操作。

---

## 安全与依赖

* 仅监听 `127.0.0.1`，不对外暴露；不引入任何第三方 Python 包。
* 前端无 JS 框架、无构建步骤，单文件 `index.html`；字体（Rajdhani / JetBrains Mono）
  从 Google Fonts 拉取，**离线时自动回退**到系统字体（微软雅黑 / Segoe UI / Consolas）。
* 日志与对话输出均做 HTML 转义，不会注入页面；对话页的 Markdown 渲染
  也遵循「先整体转义、再套用自己生成的标签」，链接仅放行 `http(s)`。
* 停止进程使用 `taskkill /F /T`，会一并结束子进程树。
* 启动器退出时通过**作业对象**自动结束自己启动的模型进程（见上文），
  外部启动的服务不受影响。

### 资源监控的实现（零依赖）

CPU / 内存 / 显存三项都不依赖 `psutil` 等第三方库：

| 指标 | 实现 |
| --- | --- |
| CPU 占用 | `ctypes` 调 `kernel32!GetSystemTimes`，相邻两次采样求 (总-空闲)/总 |
| 系统内存 | `ctypes` 调 `kernel32!GlobalMemoryStatusEx` 取物理内存总量与可用量 |
| GPU 显存 / 利用率 / 温度 | 调用 NVIDIA 驱动自带的 `nvidia-smi`（不随项目分发） |

几点设计上的取舍：

* **采样在后台线程**，`/api/sys` 只读缓存快照 —— `nvidia-smi` 启动约需几十毫秒，
  若放在请求路径上会让接口变慢（实测改为缓存后平均响应 ~16 ms）。
* **优雅降级**：没有 NVIDIA 卡或找不到 `nvidia-smi` 时，显存卡片显示
  「未找到 nvidia-smi」，CPU 与内存照常工作，不会报错。
* `nvidia-smi` **连续失败 5 次后停止重试**，避免每 2 秒拉起一个必然失败的进程。
* 快照返回**深拷贝**，外部修改不会污染监控器内部状态。
* 非 Windows 环境（无 `kernel32`）时 `ok=false`，前端显示「监控不可用」。
