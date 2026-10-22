# KVMem + ActKV · ninfer 后端

基于 [kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) 与 **ActKV**（[arXiv:2609.31395](https://arxiv.org/pdf/2609.31395)，中国科学技术大学提出）技术开发：

- **KVMem**：把长上下文稀疏化成固定窗口（`budget` 选取历史块 B + `gen_reserve` 追加工作区 R），Host/磁盘归档历史 KV，用一个固定大小的注意力窗口换来 200K/262K 的可用上下文。
- **ActKV**：面向行动的检索保持策略（LRFU 历史排序等），提升长会话中关键块的召回率。

**QQ 交流群：1040777853**

本项目仅供研究学习，稳定使用可前往kvmem原作者。

## 2026-10-22 更新：sm70 / sm75 源码支持

ninfer 后端新增 **sm75（RTX 20 系、TITAN RTX、T4）** 与 **sm70（Tesla V100）** 的构建支持。召回与驱逐保持 ActKV 不变。

### 改动范围

最小移植，不升级上游 210 个提交：

| 项 | 内容 |
|---|---|
| 新增文件 | 31 个（sm70 整套 GEMM、sm75 MMA、`pre_sm80.cuh` / `sm70_emulation.cuh` 兼容层） |
| 接入修改 | 8 处（CMake 白名单、`mma.cuh`、`memory.cuh`、sources、`QType`、2 个 codec） |
| GPU profile | 9 个（V100×2、TITAN RTX、2060S、2070、2070S、2080Ti、Quadro RTX 6000、T4） |
| ActKV | **未改动**：LRFU 公式、mean-Q 打分、sink/recent 逻辑与 10-21 版一致 |

### 构建

```bash
# RTX 20 系（sm75，CUDA 13.x 支持 compute_75）
cmake -S backends/ninfer -B build-sm75 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-sm75 --target ninfer-serve

# V100（sm70，必须 CUDA 12.x；CUDA 13 已移除 Volta）
cmake -S backends/ninfer -B build-sm70 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe"
cmake --build build-sm70 --target ninfer-serve
```

启动参数与现有配置一致，不需要 `--kvmem-query-scorer`（本地版本默认即 mean-Q cosine）。

### 验证

| 项 | 结果 |
|---|---|
| sm120a 编译 | 0 错误 |
| 64K | prefill 427.1 tok/s / decode 45.5 tok/s / 接受率 48.1% |
| 256K NIAH | 33/33、14/14、80 工具调用 |
| sm75 / sm70 运行 | **未验证**（本机只有 sm120a） |

### 块大小不可调

尝试把页大小从 64 改到 32，编译期被断言挡住：

```cpp
static_assert(kCausalPromptI8FastBc == kPagedKVPageSize);
```

页大小同时是 fast prompt 内核的 K 方向 tile，改它需要重写 MMA 迭代与共享内存预算。48 更不可行：页表索引是 `position >> 6`，要求 2 的幂。

### 文档

- [sm70/sm75 移植与块大小结论（2026-10-22）](docs/kvmem-sm75-sm70-port-20261022.md)
- [子模块 patch 恢复说明](docs/patches/README.md)

## 2026-10-21 更新：ActKV LRFU 与增量 restore，新版需自行git编译。

### 打分与驱逐

召回打分保持不变：64-token 块的 mean key 与 query 均值的余弦，prefill 结束后一次性计算。

驱逐改用 ActKV Algorithm 1（[arXiv 2609.31395](https://arxiv.org/pdf/2609.31395)）：

```
hit : KS_t = λ·KS_{t-1} + score
miss: KS_t = λ·KS_{t-1}
```

新块从 0 开始，首次命中时被当前分数点亮。与之前 EMA（`λ·KS + (1-λ)·score`）的差别在于：EMA 是加权平均，ActKV 是衰减累积和，后者让「偶尔命中但单次分高」的块活得更久。

### 增量 restore

`start_restore` 不再 `truncate(0)` 后全量重映射：

- 选中页按 id 稳定排序，同一逻辑页在重选后保持物理槽位；
- 只对变化的槽位做 H2D，也只读变化的 host 记录；
- 变化超过一半时自动回退全量，避免差异比全量还贵。

| 64K 场景 | 全量 | 增量 |
|---|---:|---:|
| restore 页数 | 576 | **124–181** |
| read | 69 ms | **14–21 ms** |
| sync | 94 ms | **20–29 ms** |
| 总耗时 | 163 ms | **33–51 ms** |

256K 时 `changed` 中位数 495/576，自动回退全量，行为与之前一致。

### 256K 召回（NIAH 33 针）

| 指标 | 结果 |
|---|---:|
| 同轮 / 终轮严格 / 并集 | 33/33 |
| 跨工具（send_alert） | 14/14 |
| 工具调用 | 80 |
| 端到端（到 final sweep） | 595.8 s |

### LongMemEval 上下文扫描

同一组实例（idx 0,1,2）、20 题、`What exactly did the user say` probe，对比 ActKV 与 EMA：

| 上下文 | ActKV | EMA |
|---|---:|---:|
| 128K | **20/20** | 19/20 |
| 192K | **18/20** | 16/20 |
| 240K | **16/20** | 15/20 |
| 合计 | **54/60** | 50/60 |

- **最佳工作区间：128K–192K**；
- 192K 开始掉分（ActKV 100% → 90%），240K 明显劣化（80%）；
- 240K 的 4 道失败里 3 道是模型拒答，不是 KV 淘汰。

### 旧更新日志

- 2026-10-18 DFlash2 × KVMem 修复：[docs/dflash2-kvmem-fixes-20261018.md](docs/dflash2-kvmem-fixes-20261018.md)
- 2026-10-19 召回传输优化（DMA 4.0 → 14.1 GB/s）：[docs/kvmem-recall-opt-20261019.md](docs/kvmem-recall-opt-20261019.md)
- 2026-10-19 重选水位提前到 80%：[docs/kvmem-reselect-high-water-20261019.md](docs/kvmem-reselect-high-water-20261019.md)

## GGUF → .ninfer 模型转换

转换工具可在Releases中下载（Python 3.11 + PyTorch/NumPy）。
`qwen3_8_27b_gguf` 配方把 Qwen3.8-27B GGUF **按自身块格式直接转换**（不做反量化），
可选加入 MTP 头、DFlash2 草稿与视觉投影器。在 `backends/ninfer` 目录下运行：

```bash
python3 -m tools.convert \
  --model /path/to/conversion-input/metadata \
  --recipe qwen3_8_27b_gguf \
  --source gguf=Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored-MTP.gguf \
  --source dflash2=Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --source vision=mmproj-Qwen3.8-27B-Q8_0.gguf \
  --components text,vision,mtp,dflash2 \
  --name qwen3.8-27b \
  --out model.ninfer
```

- `--model` 只需要基础检查点的配置与分词器（准备转换输入时生成的 `metadata` 目录）。
- `dflash2` 源支持官方 DFlash2 GGUF 或其 safetensors 目录。
- 不需要的组件可省略（例如纯文本去掉 `vision`）；`mtp` 组件要求 `-mtp` GGUF
  （`blk.64.nextn.*`）。
- CPU 转换追加 `--device cpu`。

## 快速开始

1. 下载 `.ninfer` 模型（或按上文自行转换）。
2. 双击 `workers/ninfer/start-kvmem-ninfer-dflash.bat`（DFlash2）或
   `start-kvmem-ninfer.bat`（MTP）。
   - 模型路径、启动目录与参数都在 bat 文件里，直接编辑即可。
   - 显存 16 GiB 推荐：`--max-context 262144 --kvmem-budget 36864
     --kvmem-gen-reserve 16384 --kvmem-host-mib 12288`。
3. 服务地址 `http://127.0.0.1:18200`，OpenAI 兼容 API。


## 文档

- [DFlash2 × KVMem 修复记录（2026-10-18）](docs/dflash2-kvmem-fixes-20261018.md)
- [召回传输优化：带宽基准、256K A/B 与逐次对比（2026-10-19）](docs/kvmem-recall-opt-20261019.md)
- [后端能力与启动说明](docs/multi-backend.md)
- [仓库结构与构建](docs/multi-backend-repository.md)
- [后端差异](docs/multi-backend-differences.md)
- [sm70/sm75 移植与块大小结论（2026-10-22）](docs/kvmem-sm75-sm70-port-20261022.md)
- [子模块 patch 恢复说明](docs/patches/README.md)

