# 代码级分析：`XuanzhengZhou/qwen3.8-27b-ternary-5060ti` 的代码优势

> 分析日期：2026-10-11
> 对照对象：`backends/ninfer`（`iamwavecut/ninfer-all` @ `8319e8f`）
> 方法：读他的 72 个补丁文件源码，逐项与本地引擎对照

---

## 0. 先给结论

他的代码优势**不在"更快的内核"**——你和他的三元内核思路高度重合，甚至有几处你更成熟。

他真正的代码优势是 **3 个工程性的东西**：

| # | 优势 | 性质 | 你能不能用 |
|---|---|---|---|
| **1** | **PTQ1_0 直接解码**（`__vsub4` base-3 链） | 真正的代码级领先 | ⚠️ 能用，但你不需要 |
| **2** | **代码即证据**：每个内核头注释都写着"测了什么、为什么、否掉了什么" | 工程方法论 | ✅ **最该抄的** |
| **3** | **s8 整数激活档的分层落地**（含 scratch 从 arena 分配） | 架构设计 | ✅ 部分可借鉴 |

而"听说能提速"的来源**不是这些代码**，是三元权重本身（5.9 GB vs 你 11.56 GB）。

---

## 1. 你的内核 vs 他的内核：同一套思路

### 1.1 你们用的是同一个核心技巧

**他**（`ternary_rowsplit_mma_s8.cuh:88-92`）：

```cpp
// 2-bit code byte -> four int8 values.
// TWO __byte_perm steps, and that is not decoration...
__device__ __forceinline__ unsigned ternary_s8_expand_codes(unsigned code_byte) {
    const unsigned even = __byte_perm(0x020100FFu, 0x020100FFu, code_byte);
    const unsigned odd  = __byte_perm(0x020100FFu, 0x020100FFu, code_byte >> 2);
    return __byte_perm(even, odd, 0x5140u);
}
```

**你**（`t2_prefill_i8.cuh:70-79`）：

```cpp
// One code word (sixteen 2-bit two's-complement codes, lowest k lowest) to four registers of s8.
__device__ __forceinline__ void t2_prefill_decode(unsigned word, unsigned* d) {
    constexpr unsigned kTable = 0xFFFE0100u; // 0, +1, -2, -1
    const unsigned even       = word & 0x33333333u;
    const unsigned odd        = (word >> 2) & 0x33333333u;
    d[0] = __byte_perm(kTable, 0u, even);
    d[1] = __byte_perm(kTable, 0u, odd);
    d[2] = __byte_perm(kTable, 0u, even >> 16);
    d[3] = __byte_perm(kTable, 0u, odd >> 16);
}
```

**同一件事，两种实现。** 他用 `0x020100FF` 查表 + `0x5140` 交错；你用 `0x3333` 掩码 + `0xFFFE0100` 查表。你注释里还写了"每个 code word 九条指令"（`t2_prefill_i8.cuh:19`）——**同样是精心调过的**。

`0x5140` 交错技巧你引擎里出现 7 个文件（`q4_rowsplit_storage.cuh:56`、`q5_small_t_mma.cuh:75`、`int8_g64_codec.cuh:236`…），说明这是你引擎的**通用手法**，不是他独有的。

### 1.2 bank 冲突处理：他写得更细，但手法一样

**他**（`ternary_rowsplit_mma_s8.cuh:69-74`）：

```cpp
// The 144 pitch (128 used + 16 pad) is not decoration: an ldmatrix row address must stay 16-byte
// aligned, so a plain 128 pitch would put all eight rows of one 8x16 matrix on the SAME bank
// (128/4 = 32 words, 32r mod 32 == 0). At 36 words per row the eight rows land on 4r mod 32, i.e.
// eight distinct banks, conflict-free -- the same trick the expansion plane's 48-byte pitch plays
// with 12r mod 32.
inline constexpr int kTernaryS8PlaneStride = kTernaryS8ChunkK + 16;   // 144
```

**你**（`t2_prefill_i8.cuh:48-49`）：

```cpp
// A column's 128 bytes padded to 144 so the eight columns an MMA quad reads miss each other.
static constexpr int kXStride    = 144;
```

**完全相同的 144 = 128 + 16。** 你甚至用了一模一样的数字。

---

## 2. 他唯一真正的代码级领先：PTQ1_0 直接解码

这是他**唯一一处你确实没有**的东西。

### 2.1 问题背景

PrismML 有两种三元编码：
- **PQ2_0**：2-bit 二进制补码，4 个/字节 —— **你和他都直接支持**
- **PTQ1_0**：**base-3（三进制）** 紧凑编码，更省空间

### 2.2 他的做法（`ternary_rowsplit_mma_s8.cuh:102-164`）

他**直接在共享内存里把 base-3 解码成 int8**，全程不经过 PQ2_0 中间态：

```cpp
// --- PTQ1_0 -> int8, direct (no PQ2_0 code relabelling anywhere in between) ------------------
// WHY ONE CHAIN AND NOT ONE CHAIN PER STAGE: PTQ1_0's trit for element e is (v*3)>>8 of the
// base-3 remainder recurrence v = (v*3)&0xFF applied n = e's stage index times... Four
// consecutive weights of a 4-byte qs group share that one chain: stage t's product already
// contains stage t-1's remainder in its low byte, so five multiplies by 3 produce all five
// stages of the group where five separate chains would need 1+2+3+4+5.
__device__ __forceinline__ void ternary_s8_decode_qs4(unsigned packed, std::uint8_t* dst,
                                                      int dst_step) {
    unsigned v_lo = __byte_perm(packed, 0u, 0x4140u);
    unsigned v_hi = __byte_perm(packed, 0u, 0x4342u);
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        const unsigned w_lo = v_lo * 3u;
        const unsigned w_hi = v_hi * 3u;
        v_lo                = w_lo & 0x00FF00FFu;
        v_hi                = w_hi & 0x00FF00FFu;
        *reinterpret_cast<unsigned*>(dst + t * dst_step) =
            __vsub4(__byte_perm(w_lo, w_hi, 0x7531u), 0x01010101u);
    }
}
```

**三个真正漂亮的点**：

1. **一条链而不是五条**：`(v*3)&0xFF` 的余数递推，stage t 的乘积低字节**已经含** stage t-1 的余数。五条独立链要 `1+2+3+4+5=15` 次乘法，一条链只要 **5 次**。
2. **`__byte_perm` 把四个 base-3 数字停在偶数字节**，一次 `*3` 同时推进四个数字。
3. **`__vsub4` 一条指令做四个字节的减 1**（`(xi - 1)`），四个权重一次出来。

他还在注释里写了**证明**而不是假设：

> PROVEN, NOT ASSUMED: `verify_native_ptq1_map.py` expands real weights (token_embd.weight, 64 whole rows == 327,680 grid cells) through a bit-faithful model of these two helpers placed at exactly the offsets the kernel writes, and finds 0 mismatches against ggml's `ptq1_0_trit`, against `PTQ1SimtDecodeAtom::decode_one` and against the reader's dequantizer

### 2.3 你为什么不需要它

**你的路线是"转换期解决"**（`tools/convert/ternary.py:162`）：

```python
type_matches = info.type_name in ("PQ2_0", "PTQ1_0") if kind == "PQ2_0" else info.type_name == kind
```

你的转换器**接受 PTQ1_0 输入，但在打包 `.ninfer` 时统一成 `t2_g128_fp16`（2-bit 补码）**。运行时只有一种编码，内核只写一套。

> **架构取舍**：
> - **他**：运行时多花解码指令，换取**直接用官方 GGUF**、不重新打包。
> - **你**：转换期多做一次，换取**运行时代码路径单一**（你的 `t2_rowsplit_storage.cuh:3` 只认 "2-bit two's-complement codes, four per byte"）。
>
> **两种都对。** 你的做法在长跑场景更好（运行时更快、代码更简单）；他的做法在"拿来即用"场景更好。

**顺带解释了他的坑 #5**：`NINFER_TERNARY_PTQ1_FAST=1` 不开，prefill 从 2.2k 掉到 **44.7 tok/s（≈50×）**。因为他的 PTQ1_0 路径需要这个开关才走快速解码链——**这是"运行时解码"架构的固有代价**。你的架构里**不存在这个坑**。

---

## 3. 他的第二个优势：代码即证据（最该抄）

这是他**最有价值**的地方，而且和"更快"无关。

### 3.1 每个内核头注释 = 一份实验报告

看 `ternary_rowsplit_mma_wide_t.cuh:8-38`，一个头注释写了：

- **为什么存在**：小 t 内核把 token 放外层，每 8 token 重读整个权重张量 → `ceil(T/8)` 遍
- **实测定律**：`t ≈ 12.31 ms * ceil(T/8) + 44 ms`（≤6% 误差，斜率 544 GiB/s）
- **交叉验证**：官方 runtime pp512=2087 t/s，他们只有 471–630
- **另一个佐证**：5 路并发只有 25% 效率，因为 11 token 装不进 8-token tile
- **为什么 BK=128 不是 512**：BK=512 时激活 1 KB/token/stage，48 KB 静态共享只能放 ~10 token
- **共享内存预算逐项列出**：`codes 6144 | act 34816 | lut 2048 | scales 256 = 43264 B = 42.25 KB`

### 3.2 他把"错误直觉"写进注释

`ternary_rowsplit_mma_wide_t.cuh:71-80`：

```cpp
// 4 warps (64 rows) per CTA -- the MEASURED optimum of the configurations tried on
// 2026-09-20, with the reason recorded, because the naive reading of the profiler says
// otherwise and would send the next person down the same dead end:
//   * Nsight Compute on this build: Active Warps/Scheduler 2.72 (max 12), No Eligible 77.3%,
//     12 warp-cycles per issued instruction, "Est. Local Speedup 52.49% from occupancy",
//     DRAM at only 19.7%. It reads like a pure occupancy problem.
//   * Raising occupancy to 16 warps/SM (8 warps/CTA + a 128-register cap) made everything
//     WORSE: T=64 went 52.88 -> 70.89 ms (+34%). The register cap is what did it...
```

**这段的价值**：ncu 说"提升 occupancy 能快 52%"，照做结果**慢 34%**。他把这个陷阱写下来了。

### 3.3 他甚至写清了"不抄什么"

`ternary_rowsplit_mma_s8.cuh:13-22`：

```cpp
// The recipe is the one the Prism fork uses for this exact format (read locally, not guessed):
//   E:\llama-cpp\prism-b10687\ggml\src\ggml-cuda\
//     * mmq-config-ampere.cuh:36-51  16 CASE rows for GGML_TYPE_PQ2_0, SRAM layout Q8_0.
//     * mma.cuh:946                  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.
//     ...
// What we deliberately do NOT copy: their 256 threads / occupancy=1 / I=128xJ / stream-K
// structure (that is llama.cpp's MMQ, and occupancy is measured to be the wrong knob here
// anyway), and their q8_1 per-32 activation scales (we use one scale per token over the whole
// K row, which is what keeps the epilogue a cvt+fma...)
```

**精确到行号的引用 + 明确的不抄清单。** 这是可维护性的天花板。

### 3.4 对比你的注释

你的注释**也很好**，但偏"解释这段代码在干什么"：

```cpp
// T2G128 RowSplit x int8 prefill GEMM with its own activation contract, in place of the shared
// rowsplit_a8_mma.cuh mainloop with the ternary codec. On the RTX 5060 Ti (tools/t2_a8_tile_probe.cu,
// T = 1024, L2 flushed) that one ran the text-layer shapes at 74-94 TOP/s, cuBLAS int8 at 170-184;
// this runs at 134-138 and beats cuBLAS with its conversions on every shape.
```

（`t2_prefill_i8.cuh:3-6`）

**你也写了实测数字**（74-94 vs 134-138 TOP/s、cuBLAS 170-184），质量不低。差别在**系统性**：他每个内核都写"否掉了什么"，你是零散的。

---

## 4. 他的第三个优势：s8 档的架构设计

### 4.1 scratch 从 arena 分配 + 阈值自动降级

`ternary_dispatch.h`：

```cpp
// Allocate the int8 rung's activation-quantization scratch from the caller's arena. Returns an
// EMPTY scratch below the rung's token threshold, which leaves the launch on the bf16 rungs.
// The caller owns the memory; the arena capacity must count it, which it does because
// ternary_rotation_workspace_bytes() (the capacity statement for this op family) includes it.
[[nodiscard]] TernaryS8Scratch make_ternary_s8_scratch(WorkspaceArena& workspace, std::int32_t k,
                                                       std::int32_t tokens);
```

**设计要点**：
- 低于阈值返回**空 scratch**，自动落回 bf16 档 —— 不用 `if` 分支散落各处
- workspace 容量由 `ternary_rotation_workspace_bytes()` **统一声明**，不靠调用方记得
- 注释明确警告："this op runs inside captured CUDA graphs"（阈值不能在 graph 构建和重放之间变）

### 4.2 一次旋转喂多个权重

```cpp
// Run the GEMM when the activation is ALREADY in the folded basis. Callers that feed several
// folded weights from one activation (the split GDN input projection feeds three) rotate once and
// then call this, instead of paying for -- and diverging on -- a redundant rotation per weight.
void ternary_dispatch_basis(const Tensor& x_folded, const Weight& w, Tensor& out, ...);
```

**GDN 输入投影要喂三个权重 → 只旋转一次。** 这是个真实的省时点，而且注释点明了"还会引入数值分歧"（`diverging on`）——不只是性能，还有一致性。

### 4.3 你这边对应的设计

你的 `t2_dispatch.cpp:11-30` 用**形状 switch** 选内核：

```cpp
const auto small_t = [t]() -> T2Launch {
    if (t <= 16) { return small_t_v1 ? launch_t2_small_t_mma : launch_t2_small_t_v2; }
    if (t <= 32) { return launch_t2_mma_r64_c32; }
    if (t <= 64) { return launch_t2_mma_r64_c64; }
    return launch_t2_mma_r64_c128;
};
```

**同样是按 T 分档**，而且你有 `NINFER_T2_DECODE=simt`、`NINFER_T2_SMALLT=v1`、`NINFER_T2_A8_TILE`、`NINFER_T2_I8_SMALL`、`NINFER_T2_A8_RASTER`、`NINFER_T2_A8_MIN` **六个 A/B 开关**——比他的 A/B 手段还多。

---

## 5. 逐项总账

| 维度 | 他 | 你 | 谁更好 |
|---|---|---|---|
| PQ2_0 2-bit 解码 | `__byte_perm` + `0x5140` | `0x3333` 掩码 + `0xFFFE0100` | **平手** |
| 144 = 128+16 bank pad | ✅ | ✅ 同样 144 | **平手** |
| PTQ1_0 base-3 直接解码 | ✅ `__vsub4` 单链 | ❌ 转换期转成 PQ2_0 | **他（但你有取舍理由）** |
| 运行时编码种类 | PQ2_0 + PTQ1_0（两套） | 只有 T2G128（一套） | **你** |
| PTQ1 快速路径开关坑 | ❌ 不开慢 50× | **不存在** | **你** |
| T 分档粒度 | 8/16/24/32/40/48/56/64 | ≤16 / ≤32 / ≤64 / >64 | **他更细** |
| A/B 环境开关数量 | ~4 | **6+** | **你** |
| 头注释证据密度 | **极高**（含证伪、定律拟合、行号引用） | 高（含实测数字） | **他** |
| scratch 架构 | arena + 阈值降级 | 有 `t2_a8` 分层 | **平手** |
| 多权重共享旋转 | ✅ `dispatch_basis` | — | **他** |
| 权重格式支持广度 | 三元为主 | q4/q5/q6/q8/fp8/nvfp4/三元 | **你** |
| KV 编码广度 | 少数 | int8/fp8/nvfp4/k8v4/rk8v4/rk4v4/rk4v4-e8/rk2v4-e8 | **你** |

---

## 6. 结论：对你本地 ActKV 项目的具体帮助

### 6.1 代码层面：**几乎为零**

你们的三元内核是**同一套思路的不同实现**，关键技巧（`__byte_perm` 查表、`0x5140` 交错、144 bank pad）**你已经全部有**。他唯一多的是 PTQ1_0 运行时解码，而你的转换期方案在运行时**更快、更简单、还没有他的 50× 坑**。

**不要去移植他的三元内核。**

### 6.2 真正能抄的：**注释文化**

他的内核头注释同时是**实验报告 + 证伪记录 + 决策依据**。对照你最近踩的坑——"Step 4 慢 6–7%"（后来证明是测量噪声）、"host 内存换页导致掉速"（被你正确质疑）——如果每个改动都像他那样把**判据、否掉的假设、反直觉的实测**写进代码注释，这类反复会少很多。

> **具体建议**：给你的 ActKV 改动（`accumulate_lrfu` / `commit_lrfu` / `set_retrieval_scores`）补上头注释，格式照他的：
> 1. 这个函数解决什么问题
> 2. 试过但**否掉**的方案 + 数据
> 3. 为什么是这个形状（含具体数字）
> 4. 怎么验证（oracle / 对照）

### 6.3 顺带回答"听说能提速"

**能，但来源是三元权重（5.9 GB vs 11.56 GB），不是这些代码。** 而且它省的**主要是显存，不是解码速度**：

他自己的 5060 Ti 实测（`bonsai2-27b-rtx5060ti-rtx4090.md:262-271`，散文式针prompt、`temperature 0`）：

| 上下文 | 无投机 | MTP 3 | DFlash2 7 |
|---:|---:|---:|---:|
| 9,345 | 48.4 | 74.7 | 84.8 |
| 38,116 | 44.3 | 70.0 | 78.1 |
| 194,559 | 30.8 | 46.7 | 55.2 |
| 257,342 | — | 44.2 | — |

**注意口径**：这是**散文 prompt**，接受率只有 19–37%（`同文:259-260`）。他自己的**代码 prompt** 在同一深度是 **101 tok/s**（`同文:208`）。所以这些数字**不能直接和你的召回测试对撞**——prompt 类型和投机接受率差一倍以上。

**能确定的只有一件事**：他的**裸解码（无投机）50.2 tok/s**（`同文:127`）——这是同卡、同引擎家族、最干净的可比数字。三元在 **16 GB 卡上受 422 GB/s 显存带宽约束**，5.9 GB 权重理论裸解码上限约 71 tok/s，实测 50.2 说明**已经贴着带宽跑**。

> **所以对你的结论**：换三元**不会让你的解码变快**——你的 IQ3_S（11.56 GB）同样受带宽约束，但你有 ngram（31 token/draft）+ MTP 的杠杆。三元真正的价值是**省 5.6 GB 显存**，在 98.6% 上下文利用率下等于**能开更大窗口或更大工作集**。代价是质量（98.2% 保持率）和重新打包权重。
>
> **性价比判断**：与其换权重，不如先做前面那条单行改动（`--kv-dtype rk8v4`，KV −23%）——**同样省显存、零质量重打包成本、可回滚**。

---

## 附：取证索引

| 结论 | 位置 |
|---|---|
| 他的 s8 解码（`__byte_perm` 双步） | `_ternary/repo/engine-patches/src/ops/linear/ternary/ternary_rowsplit_mma_s8.cuh:76-92` |
| 你的对应实现 | `backends/ninfer/src/ops/linear/t2/t2_prefill_i8.cuh:70-79` |
| 144 bank pad（他） | `ternary_rowsplit_mma_s8.cuh:69-74` |
| 144 bank pad（你） | `t2_prefill_i8.cuh:48-49` |
| PTQ1_0 base-3 单链解码 | `ternary_rowsplit_mma_s8.cuh:102-164` |
| 你转换期统一编码 | `backends/ninfer/tools/convert/ternary.py:162,253-274` |
| 你只认 2-bit 补码 | `backends/ninfer/src/ops/linear/t2/t2_rowsplit_storage.cuh:3` |
| 他否掉 occupancy 的实测 | `ternary_rowsplit_mma_wide_t.cuh:71-80` |
| 他引用 Prism 源码 + 不抄清单 | `ternary_rowsplit_mma_s8.cuh:13-22` |
| 你的 T 分档 | `backends/ninfer/src/ops/linear/t2/t2_dispatch.cpp:11-30` |
| 你的 6 个 A/B 开关 | `t2_dispatch.cpp:16-18`、`t2_a8.cu:61,84,336,414` |
| scratch arena 设计 | `ternary_dispatch.h`（他的） |
| 多权重共享旋转 | `ternary_dispatch.h` `ternary_dispatch_basis` |
