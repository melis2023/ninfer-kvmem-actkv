# DFlash2 直通草稿：统一量化精度（2026-10-26）

> 已汇总到 [CHANGELOG.md](CHANGELOG.md)；本文保留原始细节与过程数据。

## 1. 结论

- DFlash2 草稿的线性权重现在**统一走源工件的 GGUF 块量化**（q3km → `Q4_K` / `Q3_K`），
  不再出现同一份草稿里 Q8 与 NVFP4 混用；
- 引擎侧由 `927fa93a` 支持：动态卷积 finish 接受 `QuantLayout::GgufBlocks`，
  materialized 路径把块权重直接交给 Linear，用自己的 staging 反量化，不再要求
  `Q8_G32_FP16`；
- 仍为 BF16 的只有 `kernel_projection`、`base_kernel`、各 norm 与 selector codebook：
  conv prepare 路径没有量化内核，与精度策略无关。

**为什么这样做**：直通保留源工件的块字节，不再为第二种精度（Q8/NVFP4）另存一份，
草稿显存按 Q3/Q4 的大小计；256K 窗口下省出来的显存留给 KV 和 CUDA Graph，因此
开图也不需要 `--no-cuda-graph`。实测启动 free：DFlash2 + nvfp4 KV 为 863.8 MiB，
MTP + int8 KV 为 1,001.8 MiB。

## 2. 代码路径

| 位置 | 作用 |
|---|---|
| `tools/convert/recipes/dflash2_gguf_blocks.py` | `attention/output`、`mlp/down` 保持源 ggml 块类型，不重化为 Q8 |
| `src/ops/wrapper/dynamic_grouped_conv.cpp` | conv finish 接受 `GgufBlocks`，块权重走 Linear staging |
| `src/ops/q8/q8_dynamic_grouped_conv_add_plan.cpp` | materialized 路由的 arena 为块反量化单独预留 staging |
| `tools/convert/recipes/dflash2_import_nvfp4.py` | 另一条路径：除 QKV（Q8）与 conv（BF16）外整份草稿 NVFP4 |

## 3. q3km 工件的实际格式

用 `tools/artifact/inspect.py` 读 `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp-q3km.ninfer`：

| 张量 | 格式 | 数量 |
|---|---|---:|
| `dflash2/layers/*/attention/output` | `gguf_q4_k` | 5 |
| `dflash2/layers/*/mlp/down` | `gguf_q4_k` | 5 |
| `dflash2/feature_projection` | `gguf_q3_k` | 1 |
| `dflash2/candidate_selector/hidden_projection` | `gguf_q3_k` | 1 |
| conv `kernel_projection` / `base_kernel`、norm、codebook | `bf16` | 其余 |

整份工件的格式分布（含目标模型）：

```text
formats: bf16 626, fp32 96, gguf_iq3_s 131, gguf_iq4_xs 89,
         gguf_iq3_xxs 68, q4_g64_fp16 54, q5_g64_fp16 54,
         gguf_q4_k 50, gguf_q2_k 19, gguf_iq2_s 17, q8_g32_fp16 7,
         gguf_q3_k 7, gguf_q6_k 5, gguf_iq2_xs 9, ...
```

草稿自身没有 `q8_g32_fp16` / `nvfp4` 条目（`q8_g32_fp16` 7 个属于目标模型）。

## 4. 与 NVFP4 导入路径的对比

| 路径 | 草稿量化策略 | 例外 |
|---|---|---|
| **GGUF 直通**（`dflash2_gguf_blocks.py`） | `attention/output`、`mlp/down` 保持源块类型（Q2_K/Q3_K/Q4_K） | QKV 与 conv 保持基础配方 |
| **NVFP4 导入**（`dflash2_import_nvfp4.py`） | 除 QKV/conv 外全部 NVFP4（feature 与 selector 由 BF16 重新量化） | QKV `Q8_G32_FP16`、conv BF16 |
| 旧版混合（2026-10-23 之前） | out/down NVFP4，feature W8，selector BF16，QKV Q8 | 同一层内多种格式 |

两条现行路径都只保留**内核尚无量化的 prepare 张量**为 BF16，不再人为混用 Q8/NVFP4。

## 5. 256K 速度对比（两个启动器）

同一 950K 字符 prompt（253,031 token）、输出 256 token、temperature 0、
thinking off，RTX 5060 Ti 16 GB。prefill 取**第一遍**（第二遍是 hidden replay，
不计 token）：

| 启动器 | 草稿 | KV | B / R | prefill（第一遍） | decode | 接受率 | 总 wall |
|---|---|---|---:|---:|---:|---:|---:|
| `start-kvmem-ninfer-dflash.bat` | dflash2 draft 5 | nvfp4 | 36,480 / 16,384 | **872.5 tok/s**（290 s） | 34.8 tok/s | 20.2%（88/435） | **584.0 s** |
| `start-kvmem-ninfer.bat` | MTP draft 3 adaptive | int8 | 36,864 / 16,384 | 803.3 tok/s（315 s） | **47.5 tok/s** | **40.7%**（140/344） | 630.9 s |

说明：

- DFlash2 第一遍 prefill 快 **8.6%**，总 wall 少 **7.4%**；MTP decode 快 **36.5%**、
  接受率高 20.5 个点；
- 总 wall 里两者都含一遍 hidden replay，所以 wall 约等于两倍 prefill 时间；
- 两个启动器的 KV 精度与 B 预算不同（dflash：nvfp4 / 36,480；MTP：int8 / 36,864），
  这是仓库里现成的两份配置，对比反映**实际启动器**的差异，不是单一变量的消融；
- 两者都不需要 `--no-cuda-graph`，启动 free 分别为 863.8 MiB / 1,001.8 MiB；
- 重选严格落在 `W = B + R`，两轮都是 27 次。

选型建议：

| 场景 | 建议 |
|---|---|
| 256K 长 prompt、短输出（报告/问答） | **DFlash2 draft 5**：prefill 占绝对主导，872.5 vs 803.3 |
| 长输出、多轮对话 | MTP draft 3：接受率 40.7%，decode 47.5 vs 34.8 |
| 显存紧张 | 两者共用 q3km 工件；DFlash2 权重多约 1.1 GiB，MTP 的 free 更高 |

## 5.1 128K：同条件、无 ngram（2026-10-26）

为了排除 ngram 与 KV 精度的干扰，再跑一组严格对照：同一目标模型、`--kv-dtype nvfp4`、
B/R=36,480/16,384、`--ngram-draft-tokens 0`、126,668 token prompt，只换草稿。

| 草稿 | 第一遍 prefill | decode | 接受率 | wall |
|---|---:|---:|---:|---:|
| dflash2 + q3km | 904.8 tok/s（140 s） | 42.8 tok/s | 23.4%（137/585） | 282.8 s |
| dflash2 + q4km | 904.8 tok/s（140 s） | 43.6 tok/s | 24.2%（139/575） | 282.4 s |
| MTP + q3km（draft 3） | 904.8 tok/s（140 s） | **47.7 tok/s** | **40.8%**（140/343） | 280.9 s |

- q3 与 q4 草稿在噪声内（decode ±1.9%、接受率 ±0.8 个点），q4 多占约 0.18 GB；
- MTP decode 快 11.5%、接受率高 17.4 个点；
- 三组 prefill 完全相同（904.8 tok/s），说明 prefill 由目标模型决定，换草稿不影响；
- wall 约 281–283 s，其中约一半是 query replay；decode 仅约 6 s。

## 5.2 混合精度 vs 直通（128K，同目标）

“不开直通、用混合精度”的对照：同一份 `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` 目标 +
ModelOpt NVFP4 草稿（`I:\model\dflash2-nvfp4-modelopt`），按 `dflash2_import_nvfp4.py`
转出 `...-q3km-nvfp4.ninfer`（13.85 GB，草稿 = 12× NVFP4 + conv/norm BF16）。

条件同 5.1（126,668 token、nvfp4 KV、B/R=36,480/16,384、ngram 0、draft 5）：

| 草稿转换 | 第一遍 prefill | decode | 接受率 | wall |
|---|---:|---:|---:|---:|
| **直通 Q3_K_M**（q3km） | **904.8 tok/s**（140 s） | **42.8 tok/s** | 23.4% | **282.8 s** |
| 混合 NVFP4（import） | 844.5 tok/s（150 s） | 40.7 tok/s | 23.8% | 296.7 s |

- 直通 prefill 快 **6.7%**、decode 快 **4.9%**，接受率基本相同；
- 这台卡上 NVFP4 draft 的 A16 路径不如 GGUF 块，混合精度没有速度优势；
- 混合工件的启动 free 更低（823.3 vs 863.8 MiB）。

## 5.3 draft 长度扫描（q3km 直通，128K）

| draft | 接受率 | decode | wall |
|---|---:|---:|---:|
| 5 | 23.4%（137/585） | 42.8 tok/s | 282.8 s |
| 4 | 28.3%（135/477） | 45.8 tok/s | 288.5 s |
| **3** | **34.9%**（130/373） | **46.1 tok/s** | 288.4 s |
| MTP draft 3（对照） | **40.8%**（140/343） | **47.7 tok/s** | 280.9 s |

draft 从 5 降到 3：接受率 +11.5 个点，decode +7.7%，wall 基本不变（prefill 主导）。

## 5.4 Q8 草稿：先撞显存

`Qwen3.8-27B-DFlash2-Q8_0.gguf`（2.05 GB）直通转换后工件 14.6 GB，权重 13.0 GiB。
B/R=36,480/16,384 启动失败：

```text
FATAL | requested Engine runtime reservation requires 1729925888 bytes,
        but only 1545273344 bytes are available for runtime capacity
```

差 184 MiB，缩到 B/R=28,672/12,288（W=40,960）才能启动。因此 Q8 草稿只能作为
“更高保真换更小窗口”的选项，不适合当前 16 GB 的主力配置。

## 5.5 代码编写（128K 代码 prompt）

同一份 480K 字符的合成 Python 模块（148,774 token），要求补一个 `merge_intervals`
函数，`max_tokens 256`、greedy、ngram 0、nvfp4 KV、B/R=36,480/16,384：

| 后端 | 接受率 | decode | wall（含 query replay） |
|---|---:|---:|---:|
| dflash2 + q3km，draft 4 | **82.1%**（92/112） | **86.3–87.1 tok/s** | 350.7 s |
| MTP + q3km，draft 3 | **86.9%**（86/99） | 72.2–76.7 tok/s | 349.5–368.3 s |

- 两组第一遍 prefill 都是 **826.5 tok/s（180 s）**，完全相同（目标模型决定）；
- 代码任务的接受率远高于总结（82% vs 28%），因为输出大量复用模块里的命名和结构；
- DFlash2 draft 4 的 decode 比 MTP 快约 **20%**，接受率低 4.8 个点；
- 两组 wall 都约 350 s，其中 prefill 占 340 s 以上。

## 6. 接受率为什么低、怎么提

| 因素 | 观察 | 结论 |
|---|---|---|
| **任务类型** | 代码 82.1%、NIAH 33.5%、开放式总结 20–28% | **决定性因素**：输出越可预测，draft 越准 |
| draft 长度 | 总结任务 5 → 3：23.4% → 34.9% | 后几个位置基本白算 |
| 草稿精度 | Q4_K 23.4% vs NVFP4 23.8% | 4-bit 档位不是瓶颈 |
| ngram | 总结类输出不复用输入，命中低 | 只适合复制/改写类任务 |
| 后端 | MTP 自适应 40.8%（总结）/ 86.9%（代码） | 自适应宽度确实有效 |

### 6.1 排查：masked-draft conditioning floor（已修，无实测影响）

KVMem 下 DFlash2 的 draft 只读 `dflash_context_floor` 之后的 conditioning 行。三处设置
这个 floor 的代码不一致：

| 位置 | 原公式 | 问题 |
|---|---|---|
| `kvmem_window.cpp` 重选 | `max(recent, draft.sliding_window)` | ✅ 正确 |
| `kvmem_window.cpp` query restore | 只有 `recent`（默认 64） | ⚠️ 会隐藏 2048 窗口的大部分 |
| `planning/checkpoint_recovery.cpp` | 只有 `recent` | ⚠️ 同上 |

重选路径的注释自己写了：`recent-page span silently hides most of it and drags acceptance
down`。已把后两处改成同样的 `max(recent*64, masked_draft_ring_capacity())`。

**实测影响：没有。** 修复前后同一 128K 总结 prompt、q3km draft 4 都是
**135/477 = 28.3%**、decode 43.8–45.8 tok/s。说明本流程里 floor 已由最后一次重选设好，
restore 路径没有覆盖它；这个修复是防御性的一致性修正，不是接受率低的原因。

### 6.2 结论

低接受率来自**任务可预测性**，不是 KVMem 的 bug：

- 同一模型、同一窗口、同一 draft 4：代码 82.1%，总结 28.3%；
- 想提高接受率，优先换任务/缩短 draft；其次用 MTP 的自适应宽度；
- 代码层面的下一步：DFlash2 目前是**固定** draft 长度，没有 MTP 的 `--adaptive-mtp`。
  可以复用 `mtp_adaptive.h` 的按位置命中率估计，给 DFlash2 也做自适应宽度；按 5.3 的
曲线，开放式任务上预计能到 40% 左右，接近 MTP。

## 5.6 draft 3–7 扫描（ngram 0，鹈鹕任务）

条件：q3km、nvfp4、B/R=36,480/16,384、`--ngram-draft-tokens 0`、thinking medium、
temperature 1.0；72 token 提示词生成 HTML/SVG，`max_tokens 32768`：

| draft | decode | 接受率 | 输出 | wall |
|---:|---:|---:|---:|---:|
| 3 | 68.8 tok/s | **65.7%** | 20,171 | 294.9 s |
| **4** | **71.4 tok/s** | 54.9% | 16,720 | 235.6 s |
| 5 | 70.6 tok/s | 48.4% | 13,376 | 190.8 s |
| 6 | 67.8 tok/s | 42.4% | 22,157 | 328.5 s |
| 7 | 64.0 tok/s | 38.1% | 13,069 | 205.3 s |

- 接受率随 draft 单调下降（65.7% → 38.1%）；
- decode 在 **draft 4** 达峰值，draft 6 起被过长的草稿拖累；
- 推荐：通用 **draft 4**；要最高接受率/更短草稿环用 draft 3。

## 7. 复现

```bat
REM DFlash2
workers\ninfer\start-kvmem-ninfer-dflash.bat

REM MTP
workers\ninfer\start-kvmem-ninfer.bat
```

请求：950K 字符 filler（253,031 token），`max_tokens 256`、`temperature 0`、
`chat_template_kwargs.enable_thinking=false`。

## 8. 注意事项

- `kernel_projection` 仍必须是 BF16：`dynamic grouped conv prepare` 没有量化内核，
  recipe 会跳过它；
- QKV 在直通路径里保持基础配方的格式，NVFP4 路径里保持 `Q8_G32_FP16`（融合 attention
  输入只对 `4096x5120` 放行 NVFP4，本模型是 `6144x5120`）；
- 换 q4km 工件时同样适用：`attention/output`、`mlp/down` 会是 `gguf_q4_k`，
  `feature_projection` / selector 随源工件而定。

### 6.3 draft_bypass 的触发条件（2026-10-26 修复）

原逻辑：连续 6 轮 `accepted == 0` 就关掉草稿（切到 width=2 的无草稿路径），每 64 轮才探测一次。

问题在于判定用“连续零接受”，而不是期望收益。d4 下每轮期望是
`1 + p1 + p1·p2 + …`；即使 `p1 = 0.4` 也有约 **1.4 token/轮**，草稿仍是赚的，
但 6 连零在 `p1 = 0.4` 时每轮约 4.7%，几十轮必然触发。结果是：

- 思考阶段被误关草稿，decode 掉到普通解码的 ~40 tok/s；
- 回答开始后还要等最多 64 轮才恢复，短回答可能整段用不上 DFlash2。

修复（`decode.cpp`）：

- 用 `MtpAdaptiveSignal` 的按位置 EMA 算 `expected_tokens`；
- 只有 `expected < 1.15`（宽验证 + 草稿自身开销的盈亏线）连续 6 轮才 bypass；
- 探测间隔 64 → **约 10 轮**（strike 从 6 涨到 16），探针未回本立即重新 bypass；
- 日志改为打印 `expected=`，区分“真没用”和“被误关”。

实测（鹈鹕 SVF、draft 4、ngram 0、nvfp4）：

| | 修复前 | 修复后 |
|---|---:|---:|
| `DFLASH_BYPASS` 事件 | 1–3 | **0** |
| decode | 73.2 tok/s | 73.1 tok/s |
| 接受率 | 58.6% | 55.3%¹ |

¹ 两次输出长度不同（19,723 vs 14,357 token），接受率有采样噪声。

代码任务（148,774 token prompt）：接受率 82.1% 不变，bypass 0，无回归。