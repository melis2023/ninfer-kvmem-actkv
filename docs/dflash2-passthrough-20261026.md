# DFlash2 直通草稿：统一量化精度（2026-10-26）

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

## 6. 复现

```bat
REM DFlash2
workers\ninfer\start-kvmem-ninfer-dflash.bat

REM MTP
workers\ninfer\start-kvmem-ninfer.bat
```

请求：950K 字符 filler（253,031 token），`max_tokens 256`、`temperature 0`、
`chat_template_kwargs.enable_thinking=false`。

## 7. 注意事项

- `kernel_projection` 仍必须是 BF16：`dynamic grouped conv prepare` 没有量化内核，
  recipe 会跳过它；
- QKV 在直通路径里保持基础配方的格式，NVFP4 路径里保持 `Q8_G32_FP16`（融合 attention
  输入只对 `4096x5120` 放行 NVFP4，本模型是 `6144x5120`）；
- 换 q4km 工件时同样适用：`attention/output`、`mlp/down` 会是 `gguf_q4_k`，
  `feature_projection` / selector 随源工件而定。
