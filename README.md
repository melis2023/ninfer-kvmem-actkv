# KVMem + ActKV · ninfer 后端

基于 [kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) 与 **ActKV**
（[arXiv:2609.31395](https://arxiv.org/pdf/2609.31395)，中国科学技术大学提出）：

- **KVMem**：把长上下文稀疏成固定窗口（`budget` 历史块 B + `gen_reserve` 工作区 R），
  Host/磁盘归档其余 KV，用固定大小的注意力窗口换取 200K/262K 可用上下文。
- **ActKV**：面向行动的检索保持策略（LRFU 历史排序等），提升长会话关键块召回。

QQ 交流群：**1040777853**。本项目仅供研究学习，稳定使用请前往 kvmem 原作者。

**所有修补、测速与召回数据汇总在 [docs/CHANGELOG.md](docs/CHANGELOG.md)。**

## 当前推荐（2026-10-26）

| 项 | 推荐 |
|---|---|
| 工件 | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp-q3km.ninfer` |
| 草稿 | **q3km 直通**（Q4_K×10 + Q3_K×2，conv/norm BF16） |
| 草稿后端 | **DFlash2**，`draft 4`，`ngram 0`（agent 可用 15） |
| KV | `nvfp4`，B/R = 36,480 / 16,384（W=52,864） |
| CUDA Graph | **开**；探针校准 256 MiB，不需要 `--no-cuda-graph` |
| 驱逐 | 满窗口 100% |

```bat
--max-context 262144 --prefill-chunk 256 ^
--kvmem-budget 36480 --kvmem-gen-reserve 16384 --kvmem-host-mib 12288 --kvmem-lrfu ^
--kv-dtype nvfp4 --spec dflash2 --draft-tokens 4 --ngram-draft-tokens 0 ^
--vision --vision-residency cpu --vision-max-merged 256 ^
--default-reasoning-effort medium --default-thinking-budget 16384
```

16 GB 卡启动 free 约 780–955 MiB；B/R 按显卡调整，不要再加 `--no-cuda-graph`。

## 性能

### 128K 代码（148,774 token，补函数，256 token 输出）

| 后端 | prefill | decode | 接受率 | wall |
|---|---:|---:|---:|---:|
| **DFlash2 + q3km，draft 4** | 826.5 | **87.1 tok/s** | 82.1% | 350.7 s |
| MTP + q3km，draft 3 | 826.5 | 72.2 tok/s | **86.9%** | 349.5 s |

### 128K 总结（126,668 token）

| 后端 | prefill | decode | 接受率 | wall |
|---|---:|---:|---:|---:|
| DFlash2 + q3km，draft 4 | 904.8 | 45.8 | 28.3% | 288.5 s |
| **MTP + q3km，draft 3** | 904.8 | **47.7** | **40.8%** | 280.9 s |

### 256K（253,031 token，两个启动器）

| 启动器 | 草稿 | KV | prefill（第一遍） | decode | 接受率 | wall |
|---|---|---|---:|---:|---:|---:|
| dflash | draft 5 | nvfp4 | **872.5** | 34.8 | 20.2% | 584.0 s |
| mtp | draft 3 adaptive | int8 | 803.3 | **47.5** | **40.7%** | 630.9 s |

### 鹈鹕 HTML draft 扫描（72 token prompt，ngram 0）

| draft | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|
| decode | 68.8 | **71.4** | 70.6 | 67.8 | 64.0 |
| 接受率 | **65.7%** | 54.9% | 48.4% | 42.4% | 38.1% |

### 短 prompt / agent

| 场景 | decode | 接受率 |
|---|---:|---:|
| WebUI 鹈鹕（2 工具，thinking 209） | **96.0 tok/s** | 76.7% |
| harness（38 工具，thinking 4,826） | 68.2 tok/s | 51.9% |
| agent 9.8K（ngram 0） | 87.8 tok/s | 70.8% |
| agent 9.8K（ngram 31 仲裁） | 89.2 tok/s | 68.1% |

### KV 精度

| | `nvfp4` | `rk8v4` |
|---|---:|---:|
| 52,864 窗口 runtime | **1.62 GiB** | 1.99 GiB |
| 启动 free | **835.8 MiB** | 645.9 MiB |
| 鹈鹕 decode（draft 4） | **73.2 tok/s** | 68.6 tok/s |
| 接受率 | **58.6%** | 51.3% |

`nvfp4` 是 16 GB 卡默认；`rk8v4` 的 key 保持 INT8，理论长上下文检索更稳，代价是多约
0.37 GiB、decode 慢约 6%。

### 召回

| 测试 | 结果 |
|---|---|
| 256K NIAH，33 针 | **33/33**（同轮 / 终轮严格 / 并集） |
| 256K 跨工具 `send_alert` | **14/14** |
| 256K 三针（5% / 50% / 95%） | **3/3** |
| 128K NIAH（5% / 50% / 95%） | **3/3** |
| LongMemEval 54 题 | **54/60**（ActKV） |
| Agent 220K | **30/30** 逐轮工具召回 |

## 关键修补

完整原因、提交和验证数据见 [docs/CHANGELOG.md](docs/CHANGELOG.md)。

| 类别 | 结论 |
|---|---|
| KVMem 驱逐 | 满窗口 100%；61K 从 55 次重选降到 **0 次**，wall 82.2 → 75.8 s |
| KVMem 召回 | ActKV LRFU；LongMemEval 54/60 vs EMA 50/60 |
| KVMem 传输 | 差异 restore；64K 163 ms → **33–51 ms** |
| DFlash2 × KVMem | envelope/floor/slot 修复；越窗接受率 1.5% → **42%** |
| DFlash2 bypass | 按期望收益判定；误关草稿 **0 次** |
| ngram 仲裁 | copy 只在收益 > 1.3× 草稿时触发；鹈鹕 74.7 vs 72.1 tok/s |
| CUDA Graph | 探针校准；省 **700–800 MiB**，开图 +6% |
| NVFP4 | Blackwell 保留 scale `0x7F` → 0（仅编译验证） |
| 模板缓存 | Jinja trim 后 cache boundary 不再丢失 |
| 平台 | sm70/sm75 源码支持（未真机验证） |

## 模型转换

转换工具从 Releases 下载（Python 3.11 + PyTorch/NumPy）。在 `backends/ninfer` 下运行：

```bash
python3 -m tools.convert \
  --model /path/to/conversion-input/metadata \
  --recipe qwen3_8_27b_gguf \
  --source gguf=Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored-MTP.gguf \
  --source dflash2=Qwen3.8-27B-DFlash2-Q3_K_M.gguf \
  --source vision=mmproj-Qwen3.8-27B-Q8_0.gguf \
  --components text,vision,mtp,dflash2 \
  --name qwen3.8-27b-q3km \
  --out model.ninfer
```

- `dflash2` 源支持 GGUF、ModelOpt NVFP4、BF16、per-row FP8；
- 直通草稿用 `dflash2_gguf_blocks.py`（默认配方）；
- NVFP4 草稿走 `--override tools/convert/recipes/dflash2_import_nvfp4.py`，
  详见 [docs/dflash2-nvfp4-20261023.md](docs/dflash2-nvfp4-20261023.md)；
- `kernel_projection` 与 conv 必须保持 BF16（prepare 路径没有量化内核）。

## 快速开始

1. 下载 `.ninfer` 工件，或按上文自行转换。
2. 双击 `workers/ninfer/start-kvmem-ninfer-dflash.bat`（DFlash2）或
   `start-kvmem-ninfer.bat`（MTP）。模型路径和参数都在 bat 里。
3. 服务地址 `http://127.0.0.1:18200`，OpenAI 兼容 API。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | **所有修补、测速、召回总览** |
| [docs/dflash2-passthrough-20261026.md](docs/dflash2-passthrough-20261026.md) | q3km 直通、draft 扫描、bypass/floor 细节 |
| [docs/dflash2-kvmem-fixes-20261018.md](docs/dflash2-kvmem-fixes-20261018.md) | DFlash2 × KVMem 首次修复 |
| [docs/kvmem-reselect-high-water-20261019.md](docs/kvmem-reselect-high-water-20261019.md) | 驱逐水位与 issue #6 |
| [docs/kvmem-actkv-lrfu-20261020.md](docs/kvmem-actkv-lrfu-20261020.md) | ActKV LRFU 与 LongMemEval |
| [docs/cuda-graph-calibration.md](docs/cuda-graph-calibration.md) | Graph 探针与启动数据 |
| [docs/multi-backend.md](docs/multi-backend.md) | 后端能力与启动说明 |
| [docs/multi-backend-repository.md](docs/multi-backend-repository.md) | 仓库结构与构建 |
