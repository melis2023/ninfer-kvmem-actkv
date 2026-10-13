# 分析：`XuanzhengZhou/qwen3.8-27b-ternary-5060ti` 对本项目（KVMem + ActKV）的价值

> 分析日期：2026-10-11
> 对象：<https://github.com/XuanzhengZhou/qwen3.8-27b-ternary-5060ti>
> 本地对照：`backends/ninfer`（submodule 固定 `8319e8f`，origin `iamwavecut/ninfer-all`）
> 方法：浅克隆全树（`I:\dsh\html\_ternary\repo`）+ 与本地引擎逐文件哈希比对

---

## 0. 一句话结论

这个项目和你本地是**同一个引擎家族（ninfer）的姊妹分支**：它的 72 个补丁文件里，**44 个在你本地树里有同名文件**。

但结论可能和"听说能提高速度"相反：

- **它的内核补丁对你几乎没用** —— 你本地引擎（`ninfer-all`）已经内置了它绝大部分的东西，包括它宣称"为 sm_120 修的那个 bug"。
- **真正值钱的是三样非内核的东西**：① KV 编码格式的实测账本；② "PPL ≤ +0.05" 的质量门与 20+ 条证伪清单；③ 一份 5 条 💀 的"静默失败"清单。
- **它唯一的"提速"来源是三元权重 + 它那套更省的 KV 编码**，不是补丁本身。而其中**换 KV 编码这一条，你本地引擎已经支持，只是你的启动脚本没在用**。

---

## 1. 这个项目是什么

| 项 | 值 |
|---|---|
| 模型 | Ternary-Bonsai-2-27B（PrismML 对 Qwen3.8-27B 的三值量化，1.72 bit/weight，~9.3× 压缩） |
| 权重 | 5.9 GB（PQ2_0 GGUF 7.21 GB；`.ninfer` 组装件 10.5 GB） |
| 硬件 | **RTX 5060 Ti 16GB，sm_120a，36 SM**（和你同一张卡） |
| 系统 | Ubuntu 24.04，驱动 580.178.04，CUDA 13.3 |
| 引擎 | `paicat1/Bonsai-27B-NInfer`（`engine-main`，基线 `f90e30c1`）+ **72 文件补丁** |
| 规模 | **143 轮迭代**（2026-10-05 → 10-11），26 轮内核快照 r7–r42 |
| 质量账 | 整套栈 **ΔPPL +0.0217（+0.25%）**，门限 +0.05（用掉 43%） |

架构：`dsh 适配器 :18801 → 混合路由 :18800 → ninfer 引擎 :18787`，watchdog 自愈。

---

## 2. 血缘关系：为什么"补丁"基本不可直接用

| | 你本地 | 这个项目 |
|---|---|---|
| 引擎仓库 | `iamwavecut/ninfer-all` @ `8319e8f` | `paicat1/Bonsai-27B-NInfer` @ `f90e30c1` |
| 血缘 | Neroued/ninfer 的 RTX 30/40/50 合并线 | Neroued/ninfer 的 **Windows 移植**（参照 v1.0.5/v1.0.6） |
| 补丁文件数 | — | 72 |
| **同名文件** | **44 / 72** | |
| **逐字节相同** | **0 / 44** | |

44 个同名文件**一个都不相同**，而且**多数是你这边更大**：

| 文件 | 项目版 | 你本地版 |
|---|---:|---:|
| `serve/serve_options.cpp` | 22.2 KB | **103.3 KB** |
| `serve/http_server.cpp` | 27.4 KB | **75.2 KB** |
| `runtime/engine/engine.cpp` | 26.8 KB | **42.6 KB** |
| `ops/wrapper/gdn_input_proj.cpp` | 71.0 KB | **79.5 KB** |
| `ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp` | 5.4 KB | **19.1 KB** |

**判断：你的引擎是更新、更全的那一支。** 它是"往上游追"，你是"合并线 + KVMem/ActKV 增量"。**不要往回移植内核代码。**

### 2.1 它的头号 sm_120 修复，你已经有了

项目文档 `docs/performance/bonsai2-27b-rtx5060ti-rtx4090.md:27-36` 说：

> **GDN gating：36 SM 装不下协作网格。** 路由表按最少 82 SM 设计（`kMinSupportedSmCount`）。5060 Ti 上表格 split-k 超出常驻 CTA 预算，第一次预填就崩 `cudaErrorCooperativeLaunchTooLarge`。修复：`fitted_schedule` 沿 split-k 阶梯（32,16,8,4,2,无 split）下探直到网格可常驻。

你本地 `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp`：

- `kMinSupportedSmCount = 82`（第 98 行）—— 同样存在
- `device_sm_count()`（第 104-120 行）—— **运行时真实读取 SM 数**（你卡上返回 36）
- `resolve_from_routes()`（第 499-511 行）—— 逐条往后找"在这台设备上合法"的路由：

```cpp
// The route table is tuned on 82 SMs. A cooperative launch needs its whole grid resident, so a
// route whose grid exceeds what this device holds is not merely slow, the driver rejects it. On
// a device with fewer SMs the covering route hands over to the first later route that is legal
// there: later routes split K less, so they launch fewer CTAs, and the last is unsplit and needs
// no residency.
```

这就是它 `fitted_schedule` 的等价物，**只是实现方式不同**。它那边缺这个才崩，你这边不会。

> ✅ **可验证的结论**：这个项目最出名的那条 sm_120 修复，对你**没有增量价值**。

---

## 3. 你本地已经有的（所以别抄）

逐项 grep 本地树确认：

| 能力 | 项目 | 你本地 | 证据 |
|---|---|---|---|
| 三元 T2 内核 | ✅ | ✅ **已有** | `src/ops/linear/t2/`（21 文件：`t2_a8.cu`、`t2_dispatch.cpp`、`t2_rowsplit_gemm_mma.cu`…）；`weight.h:21` `T2_G128_FP16 = 9` |
| Bonsai 转换配方 | ✅ | ✅ **已有** | `tools/convert/ternary.py`（`bonsai2_27b_ternary`）；`tests/convert/test_ternary.py` |
| 5060 Ti 三元实测 | ✅ | ✅ **已有** | `docs/performance/bonsai2-27b-rtx5060ti-rtx4090.md`（你自己的树里！） |
| `rk8v4` KV | ✅ | ✅ 已有 | `serve_options.cpp:61` → `RotatedInt8KeyInt4ValueGroup64` |
| `rk4v4` / `rk4v4-e8` / `rk2v4-e8` | 部分 | ✅ **更全** | `serve_options.cpp:58-61`；`paged_kv_storage.h:71-99` |
| nvfp4 / k8v4 KV | ✅ | ✅ 已有 | `paged_kv_storage.h:67-68` |
| DFlash2 / MTP / ngram / adaptive-mtp | ✅ | ✅ 已有 | `serve_options.cpp:1141-1153` |
| prefix cache | ✅ | ✅ 已有（56 文件） | `src/runtime/prefix_cache/`、`src/runtime/engine/context_cache/` |
| `--kv-lease-growth` | — | ✅ 已有（默认关） | `docs/serving.md:1836` |
| `--recover-invariant-failures` | ✅ | ✅ 已有（默认关） | `docs/serving.md:1743` |
| HTTP chunked + `TCP_NODELAY` | ✅ | ✅ 已有 | `http_server.cpp:316`、多处 `set_chunked_content_provider` |
| **FP8 GDN 状态镜像** | ✅ | ❌ **没有**（只有 fp16） | grep `state_fp8` 命中 0；`types.h:626` 只有 `gdn_state_fp16` |

**所以"听说能提高速度"的真相**：那个速度来自**三元权重**（5.9 GB vs 你 11.56 GB）+ **它那套 KV 编码**，不是补丁。你已经有全部三元工具链，可以自己复现——但**换模型 ≠ 换引擎补丁**。

---

## 4. 真正对 ActKV / KVMem 有用的 5 件事

### 4.1 ⭐ KV 编码账本 —— 最高价值，且**你今天就能试**

项目实测（`README.md:97`、`README.md:219-223`、`README.md:258-261`），单位 = 字节 / token / KV head：

| 编码 | 字节/token/head | 相对 int8 | 实测质量代价（其引擎语料） |
|---|---:|---:|---|
| `bf16` | 1024 | +94% | 基线 |
| **`int8`（你现在用的）** | **~528** | — | — |
| `rk8v4` | **408** | **−23%** | 其默认档 |
| `rk4v4-e8` | **280** | **−47%** | PPL 5.631 → **5.650** |
| `rk2v4-e8` | **216** | **−59%** | PPL 5.631 → **5.820**；DFlash2 接受率 54.4% → 51.8%，解码 −4% |

**为什么这对你特别重要**：你的 KVMem 已经跑到 **258,467 / 262,144 = 98.6%** 峰值。KV 是唯一的硬约束，**少 23% 的 KV 直接等于多 23% 的上下文或工作集余量**。

**而且你本地引擎已经全部支持**：

- `src/models/qwen3_5/program/storage/kvmem_window.cpp:69-80` —— KVMem 的 page layout **已经为这些编码注册**：
  `ninfer.rk8v4-k64-v32.page.v1` / `rk4v4-lloyd-k64-v32.page.v1` / `rk4v4-e8-k64-v32.page.v1` / `rk2v4-e8-root-k64-v32.page.v1`
- `src/serve/serve_options.cpp:1838` —— 校验串明确列出 KVMem 允许
  `INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8`

**但你的启动脚本写的是**：

```
--kv-dtype int8
```

> 🎯 **这是本项目给你的、成本最低、收益最明确的一个实验：把 `--kv-dtype int8` 改成 `--kv-dtype rk8v4`，重跑同一套 256K/33 针召回测试。** 预期 KV 占用 −23%，质量代价（按它的账本）约 +0.3% PPL。若召回不掉，再试 `rk4v4-e8`（−47%）。

### 4.2 质量门：`ΔPPL ≤ +0.05`

他们 143 轮每一条改动都过同一个门，整套栈只用掉 **43%** 的预算（+0.0217 / +0.05）。

你的现状：改了 A9 延迟 LRFU、改过 ngram 档位、改过 budget，判据一直是**召回测试**。召回是"必要不充分"——它在项目自己的坑表里被列为最贵的一类（见 4.3）。

> **建议**：给 ActKV 的每次改动配一条 `ninfer-perplexity` 门（你本地有 `apps/perplexity`），至少用 `--quick`。否则"召回没掉"不等于"质量没掉"。

### 4.3 ⭐ "静默失败"清单 —— 直接对应你的 KVMem

`docs/linux-port/docs/04-已知坑总表.md` 把最贵的坑按代价排序，前 5 条全是 **"HTTP 200 + 日志零报错 + 答案看着正常"**：

| # | 坑 | 症状 |
|---|---|---|
| 1 💀 | **题面超池 ⇒ 中段内容静默丢失** | HTTP 200、`finish_reason=stop`、**日志零报错**；根因至今未定位 |
| 2 💀 | **客户端 `max_tokens` > 池容量** | **打崩 worker 且不自愈**：此后全 503，而 `/v1/models` 仍返回 200 |
| 3 💀 | `/v1/models` 200 ≠ 可服务 | 判活一律用 `/health` + 真请求 |
| 4 💀 | **五个 ring 开关缺一不可** | 只设 WINDOW+REUSE 时，报 99.9% 缓存命中但**长题面答错（中段看不见）**，零报错 |
| 5 💀 | PTQ1 fast-rung 开关不开 | prefill 44.7 tok/s vs 2.2k（≈50×），无报错，看起来像"模型就是慢" |

**坑 #1 和 #4 描述的现象，和"KVMem 淘汰掉了该留的块"完全同构**——都是中段消失、不报错、答案流畅。

> 🎯 **建议**：你的 33/33、39/40 召回测试是**正控**。按他们的方法论，每条改动还要配**负控**（故意让池溢出、故意把 `max_tokens` 开到超池，确认告警**恰好**响一次；确认没动过的地方告警**一次都不响**）。
>
> 坑 #2 对你尤其现实：你跑 `--max-context 262144` + KVMem 池 53248，客户端如果发一个 `max_tokens` 很大的请求，按他们的记录会**打崩 worker 且不自愈**。你本地有 `--recover-invariant-failures`（默认关）——值得打开。

### 4.4 长预填是单管、且已被硬件饱和

- 预填速率 **1.52k tok/s @8K → 0.53k @256K**，**与并发路数无关**："预填并行化"不存在，只有排队顺序能优化（`GOAL:145`）
- v113 定论：注意力 **39.4 TFLOP/s = f16 峰值 78%**，GEMM **≈ int8 峰值 81%**；250K 外推 460 s vs 实测 458 s（差 0.4%）⇒ **没有优化空间**
- 他们的 250K 冷预填最好成绩 **458.2 s**；5060 Ti 上 257342 token 档 **507 tok/s**（MTP）

对照你的实测：池满后预填 **310–400 tok/s**，从 80k 到 255k 基本平。

> **解读**：他们的结论说明，长预填的瓶颈在**搬运的字节数**，不在内核。你比他们慢的那一截，最可能的原因就是 **int8 KV 比 rk8v4/nvfp4 多搬 23%–50% 的字节**。这又一次指向 4.1。

### 4.5 证伪清单 —— 20+ 条，避免重走

来自 `CHANGELOG.md:92-97`、`GOAL-对照总表`、`优化总纲-一页版`：

| 试过但**没用**的 | 结果 |
|---|---|
| f16 累加注意力 | 孤立测 2.00× 快，但 250K 预填只 −4.45%（~85% tile 时间在非张量路径） |
| FP8 KV | 只有 −4.9%（smem 墙：99.0 KiB/block vs 100 KiB/SM，禁止 K/V 双缓冲） |
| 字粒度 LUT | **零提速**（57.26 → 52.38 tok/s） |
| 192-token tile | 3 种配置全更慢（495/705 vs 785）—— 寄存器文件边界 |
| 提高每 SM warp 数 | occupancy 上限只值 **~2%** |
| CUDA Graph 消除 launch | 只在 C≤8 划算（+6.9%/+5.9%），图池代价 2.5–3 GB |
| 自适应 verify 宽度 | 端到端 **−26~53%**，接受率 0.80→0.37 ⇒ 默认关 |
| 更宽的 verify（d7 > d3） | C=16 **−17%**、C=32 **−27%** |
| 多路预填改善长上下文 | 256K 档不可达（最便宜组合仍差 0.37 GB） |
| 预填 tile 放大 / GEMM DRAM 受限 / T=128 两趟权重 | 全部证伪 |
| 工具 schema 无损压缩 | 零余量（2608 → 2608） |
| 步探针耗时 = 设备时间 | probe 有 **×3.71 观察者效应** |
| KVMem 代码级移植 | "不可行且不值得"——上游已自带 Device/Host KV |

> 最后一条对你有点刺：他们评估过**移植 KVMem**并判定不值得（`落地实施方案:167-168`）。这不影响你的路线（你已经在 KVMem 上做了 ActKV），但说明**上游认为自己的 Device/Host KV + prefix cache 已经够用**——值得你回头看一次差距在哪。

---

## 5. 不要抄的

| 项 | 原因 |
|---|---|
| 三元 T2 内核补丁 | 你已有；且需要重新量化权重 |
| FP8 GDN 状态镜像 | 你只有 fp16 版；收益是**并发路数**（42→60），你是单用户，无感 |
| 多路并发那一整套 | 62 路、聚合 799 tok/s、admission ladder —— 全是多租户收益 |
| L1–L4 适配器层 | 它是 `dsh_adapter.py` 外挂；你的架构不同（引擎内 prefix cache + KVMem） |
| Linux/systemd 运维 | 你是 Windows `.bat` 单用户 |
| sm_120 `fitted_schedule` | 你已有等价物（`resolve_from_routes`） |

---

## 6. 建议的下一步（按性价比排序）

| # | 动作 | 成本 | 预期收益 | 判据 |
|---|---|---|---|---|
| 1 | 把启动脚本 `--kv-dtype int8` → `rk8v4`，重跑 256K/33 针 | **单行改动** | KV −23% | 召回不掉 + PPL Δ ≤ +0.05 |
| 2 | 若 #1 通过，再试 `rk4v4-e8`（−47%） | 单行 | KV −47% | 同上（预期 PPL +0.019 量级） |
| 3 | 打开 `--recover-invariant-failures` | 单行 | 防"打崩不自愈" | 故意超池，确认自愈 |
| 4 | 给 ActKV 改动加 PPL 门 | 1 次跑分 | 防止静默质量损失 | `ninfer-perplexity --quick` |
| 5 | 配负控（正控你已有） | 半天 | 区分"修好了"和"看起来没事" | 溢出告警恰好响一次 |
| 6 | 读 `04-已知坑总表.md` 对照 KVMem 逐条排查 | 1 小时 | 排雷 | — |

---

## 7. 一句话总结

> 这个项目**不是一个可以"装上去就变快"的补丁包**——它的内核补丁对你没有增量，因为你的引擎更新、更全，连它最出名的 sm_120 修复你都已经有了。
>
> 它真正的价值是**一份 143 轮、带质量门和证伪清单的实测账本**。而账本里对你最有用的一页是：**KV 编码格式能省 23%–59% 的 KV**，你本地引擎已经全部支持，只是启动脚本还在用最贵的 `int8`。

---

## 附：本地取证索引

| 结论 | 本地文件:行 |
|---|---|
| 引擎 origin | `backends/ninfer` → `iamwavecut/ninfer-all` @ `8319e8f` |
| 三元内核已存在 | `backends/ninfer/src/ops/linear/t2/`（21 文件）、`src/core/weight.h:21` |
| 三元转换配方 | `backends/ninfer/tools/convert/ternary.py` |
| 5060 Ti 三元实测 | `backends/ninfer/docs/performance/bonsai2-27b-rtx5060ti-rtx4090.md` |
| sm_120 路由回退 | `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:98,104-120,499-511` |
| KVMem 已注册 rk8v4 等 | `src/models/qwen3_5/program/storage/kvmem_window.cpp:69-80` |
| KVMem + rk8v4 校验放行 | `src/serve/serve_options.cpp:1838` |
| `--kv-lease-growth` | `docs/serving.md:1836` |
| `--recover-invariant-failures` | `docs/serving.md:1743` |
| KVMem 关闭原生 context cache | `src/serve/serve_options.cpp:1864-1866` |
| 当前启动脚本 | `workers/ninfer/start-kvmem-ninfer.bat:162`（`--kv-dtype int8`） |
