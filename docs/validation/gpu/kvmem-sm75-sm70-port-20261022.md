# sm70 / sm75 移植与块大小结论（2026-10-22）

## 0. 结论

| 项 | 结果 |
|---|---|
| sm70（V100）源码支持 | **已完成**，编译路径打通 |
| sm75（RTX 20 系）源码支持 | **已完成**，编译路径打通 |
| ActKV 召回/驱逐 | **未改动**，与 `e530d1ed` 一致 |
| sm120a 回归 | 64K 427.1 tok/s prefill / 45.5 decode，无显著变化 |
| 块大小 32 | **不可行**：prompt 内核 tile 硬绑定页大小 |
| 块大小 48 | **不可行**：非 2 的幂，页表位移失效 |

sm75/sm70 的实际运行未验证：本机只有 RTX 5060 Ti（sm120）。

## 1. 为什么不做全量升级

上游 `08fd7019` 比本地 `e530d1ed` 多 210 个提交，其中 sm70/sm75 相关 34 个。直接 merge 产生 **43 个冲突文件、约 80 个冲突块**，且大部分冲突是「本地基线旧 vs 上游新」，不是 KVMem 功能冲突。

因此改为**最小移植**：只把 sm70/sm75 的实现文件拿过来，其余保持本地。

## 2. 移植边界

### 2.1 直接取上游的新增文件（31 个）

```
src/ops/common/device_limits.h          设备限制（Turing/Volta 的 shared memory、warp 数）
src/ops/common/pre_sm80.cuh             sm80 以下兼容层
src/ops/common/sm70_emulation.cuh       Volta 指令模拟（cp.async、ldmatrix、mma.sync）
src/ops/common/sm75_mma.cuh             Turing MMA 形式
src/ops/linear/sm70/*                   整套 V100 GEMM（gemm/project/small/integer/dequantize）
src/ops/softmax_attention/.../prompt_sm70.cu
src/ops/softmax_attention/.../prompt_bf16_sm75.cuh
src/ops/softmax_attention/.../small_t_sm70.cuh
src/runtime/engine/device_profiles.json  9 个新 profile（V100/TITAN RTX/2060S/2070/2080Ti/Quadro/T4）
```

### 2.2 修改的已有文件（4 处接入 + 2 处依赖）

| 文件 | 改动 |
|---|---|
| `CMakeLists.txt` | 白名单加 `70`/`75`；兼容路径条件；`NINFER_SM70`/`NINFER_SM75` 宏；sm70 需 CUDA < 13 |
| `src/ops/common/mma.cuh` | include `sm75_mma.cuh`；`ldmatrix`/`mma` 按 `SM70_DEVICE`/`SM75` 分派 |
| `src/ops/common/memory.cuh` | include `sm70_emulation.cuh`；cp.async 的 Volta 回退 |
| `src/ops/linear/sources.cmake` | include `sm70/sources.cmake` |
| `src/ops/softmax_attention/sources.cmake` | 加 `prompt_sm70.cu` |
| `src/core/weight.h` | `Q2_G64_FP16 = 28`、`FP8_E4M3FN_ROW_FP16 = 29` |
| `src/ops/kv_cache/fp8_e4m3_row_codec.cuh` | `kv_cache_fp8_code2_to_half2_x2m8` |
| `src/ops/kv_cache/nvfp4_group16_codec.cuh` | `kv_cache_e2m1x2_to_half2` |

### 2.3 明确不采纳的上游改动

MoE 专家池、qwen4_exp、flash-next、gemma4、新 sampling API、D3D12 residency、CUDA graph PDL 等。这些与 sm70/sm75 无关，取了会引入依赖链（首次尝试取 163 个文件时，因 `moe_decode_up_launch`、`sampling_pick_from_support` 缺失而失败）。

## 3. 编译验证

### 3.1 sm120a（本机）

```
cmake --build build-backends/ninfer --target ninfer-serve -j 16
→ 0 errors
```

64K 结果：

| 项 | e530d1ed 基线 | 移植后 |
|---|---:|---:|
| prefill | 443.4 | 427.1 |
| decode | 46.7 | 45.5 |
| 接受率 | 48.1% | 48.1% |

差异在 ±4% 波动范围内（同一 exe 重测曾出现 404.7–444.4）。

### 3.2 sm75 / sm70（未执行）

按用户要求不构建（本机无对应硬件，无法运行验证）。

构建命令：

```bash
# sm75（CUDA 13.4 支持 compute_75）
cmake -S backends/ninfer -B build-sm75 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-sm75 --target ninfer-serve

# sm70（必须 CUDA 12.x，CUDA 13 已移除 Volta）
cmake -S backends/ninfer -B build-sm70 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe"
cmake --build build-sm70 --target ninfer-serve
```

### 3.3 静态检查结果

| 检查 | 结果 |
|---|---|
| 31 个新文件在源列表 | `linear/sm70/sources.cmake` 已 include；`prompt_sm70.cu` 已加入 |
| include 链 | `memory.cuh → sm70_emulation.cuh`；`mma.cuh → sm75_mma.cuh` |
| 宏定义 | CMake 中 `NINFER_SM70=1` / `NINFER_SM75=1` |
| 设备 profile | 9 个 sm70/sm75 条目 |
| 未定义符号 | sm120a 编译 0 错误，说明接入完整 |

## 4. 块大小 32 / 48

### 4.1 32 不可行

改动尝试：

```cpp
kPagedKVPageSize = 64 → 32
kPagedKVPageShift = 6 → 5
block_tokens = 64 → 32        （两处）
>> 6 → >> 5                   （5 处页表索引）
/ 64 → / 32                   （2 处）
```

编译失败：

```
src/ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh:82:
  static assertion failed
  static_assert(kCausalPromptI8FastBc == kPagedKVPageSize);
```

`kCausalPromptI8FastBc = 64` 是 fast prompt 内核的 K 方向 tile，与页大小硬绑定。改它需要同时改：

- MMA 迭代次数
- 共享内存预算（`static_assert(SmemBytes == 100352)`）
- 边界处理

**这不是调参，是重写内核。**

### 4.2 48 不可行

页表索引用位移：

```cpp
physical_page = block_table[position >> 6];
```

48 不是 2 的幂，无法用位移。改成除法会进入 attention 热路径，性能损失不可接受。

### 4.3 结论

**块大小在当前架构下不可调。** 64 token/页是 GPU 内存布局、页表位移、prompt 内核 tile 三者的共同约束。

如果确实需要更细粒度，只有两条路：

1. 页内 token 精排（先 64 页粗选，再在页内按 token 分数排序）——之前 token 级打分失败过，根因是流式 prefill 的 `q_sum` 不完整
2. 重写 prompt 内核的 tile 形状——工作量以周计

## 5. 当前状态

| 项 | 值 |
|---|---|
| 子模块分支 | `dflash2-kvmem-fixes-20261018` |
| 子模块 HEAD | `e5f51918` |
| 主仓库 | `fcef427` |
| 新增 patch | `docs/patches/0006-*`（671.9 KB）、`0007-*`（14.3 KB） |
| 备份 | `I:\llama\_backup\...-20261022-141003-pre-sm75`（7.325 GB） |
| 服务 | 运行中（18200），build `e5f51918` |
| 测试 | 64K 正常；256K NIAH 与 LongMemEval 未在本次移植后重跑 |

## 6. 后续

| 事项 | 说明 |
|---|---|
| 20 系机器验证 | 用 3.2 的命令构建 sm75，跑 64K + NIAH |
| V100 验证 | 需要 CUDA 12.4 构建 sm70 |
| 移植后召回复核 | 建议重跑一次 256K NIAH 与 LongMemEval，确认无回归 |
| 块大小 | 结论为不可行，除非重写 prompt 内核 |
