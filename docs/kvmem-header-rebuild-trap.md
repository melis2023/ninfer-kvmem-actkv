# 构建陷阱：改 `kvmem_store.hpp` 后必须强制重编

## 症状

ninfer server 在 250k 上下文下崩溃：

| 时间 | 退出码 | 崩溃位置 |
|---|---|---|
| 22:00 | `0xC0000005` | `VCRUNTIME140.dll+0x1dc10`（memcpy） |
| 22:17 | `0xC0000005` | `infer-kvmem-server.exe+0x46eaec` = **`KvMemStore::accumulate_lrfu+0x16C`** |

同一 prompt 在 **ActKV 关闭**时能完整跑完（254,557 token / 689 s）。崩溃点总在
**prefill 结束、第一次检索打分**时（两臂 prefill 速率曲线逐点相同，ON 臂在 ~350 s 死亡）。

## 定位过程

1. Windows 事件日志给出崩溃偏移：`Application Error` 事件 → `Fault offset: 0x46eaec`。
2. 用 `/MAP` 重链（**只影响链接、不改变代码生成**，所以地址与崩溃二进制一致）：

   ```powershell
   scripts\windows\build-backends.ps1 -Backend ninfer -CudaPath <cuda> -Jobs 10 -CMakeArgs '-DCMAKE_EXE_LINKER_FLAGS=/MAP'
   ```

3. 在 `infer-serve.map` 里按 `Rva+Base = 0x140000000 + 0x46eaec` 二分查找最近的符号
   （MAP 格式：`段:偏移  符号  Rva+Base  对象`，基址见文件头 `Preferred load address`）。
4. 结果落在 `?accumulate_lrfu@KvMemStore@kvmem@@...` 内，`+0x16C`。
5. `dumpbin /DISASM` 全量反汇编（515 MB）后定位到崩溃指令：

   ```asm
   14046EA9D: lea  rbx,[rcx+180h]      ; rbx = &lrfu_scratch_  (this+0x180)
   14046EAA4: mov  rax,[rbx]           ; begin
   14046EAAD: mov  [rbx+8],rax         ; clear(): end = begin
   14046EAE2: mov  rdx,[rbx+8]         ; rdx = end
   14046EAE6: cmp  rdx,[rbx+10h]       ; end vs capacity
   14046EAEC: movsd [rdx],xmm0         ; ← 崩溃：写野指针
   ```

## 根因：增量构建的 ABI 错配（ODR 违规）

`KvMemStore` 的成员布局变了，但**只有 `kvmem_store.cpp.obj` 被重编**：

| 文件 | 时间戳 | 状态 |
|---|---|---|
| `kvmem_store.hpp`（加了 `lrfu_last_round_`/`lrfu_round_seen_`，随后又加 `lrfu_scratch_`） | 21:25 / 22:08 | 已改 |
| `kvmem/src/host/kvmem_store.cpp.obj` | 21:26 / 22:09 | ✅ 重编 |
| `backends/ninfer/.../memory_statistics.cpp.obj`（**构造** `KvMemStore` 的地方） | **20:24** | ❌ 未重编 |
| `backends/ninfer/.../kvmem_window.cpp.obj` | **20:24** | ❌ 未重编 |

构建日志印证：那两次"重编"都是 `[2/4]`，**只编译了 1 个 obj**，完全没碰 ninfer 侧包含该头的 TU。

于是：ninfer 的 TU 按**旧 `sizeof(KvMemStore)`**（少一个 24 字节的 `std::vector`）分配对象，
`accumulate_lrfu` 却按新偏移 `0x180` 访问 `lrfu_scratch_` → **写到对象边界之外** → 崩溃。
这也解释了为什么只有 ActKV 打开才崩：只有它会读那个成员。

## 更深一层：这些 TU 为什么根本没被重编

本文原来的解释是"`kvmem/` 是 `-DNINFER_KVMEM_SOURCE_DIR` 引入的**外部**目录，它的头文件变化没有被
正确跟踪"。**2026-10-15 查清后，这个解释只对了一半，真正的机制影响面更广。**

`cl.exe` 用**系统语言**输出 `/showIncludes` 行（本机是 `注意: 包含文件:`）。CMake 4.3 探测这个前缀后
**双重编码**写进 `CMakeFiles/rules.ninja`：

```
正确的 UTF-8：      e6 b3 a8 e6 84 8f 3a 20 e5 8c 85 ...
rules.ninja 里却是：e5 a8 89 e3 84 a6 e5 89 b0 3a 20 e9 8d 96 ...   ← 把 UTF-8 当 GBK 解码后重编码
```

ninja 拿这个乱码去匹配 cl 的输出，永远匹配不到，于是**每个 C++ 对象的依赖列表都是空的**：

```
ninja -t deps .../silu_mul.cpp.obj : #deps 0        ← 零依赖
ninja -t deps .../moe_route.cu.obj : #deps 201      ← CUDA 走 nvcc 的 depfile，正常
```

所以不是"外部目录没被跟踪"，而是**所有 C++ 头依赖都没被跟踪**；`kvmem/` 只是最容易撞上的那一个。
同一个缺陷后来以另一种形式再次复现：给 `ninfer::KvmemOptions` 加一个 float，`main.cpp`（旧布局）
写 `conf_start_tokens = 0`，`serve_options.cpp`（新布局）偏移 4 字节去读，读到相邻 `conf_scale_up`
的 `1.75f` 位模式（1071644672，恰好是 64 的倍数且 > 36864），于是**只传 `--kvmem-budget` 就报
`--kvmem-conf-budget must not exceed --kvmem-budget`**。

## 修复

`scripts/windows/build.ps1` **一直有正确的处理**：探测 cl 的真实前缀（不按语言硬编码，而是从
`/showIncludes` 输出里正则提取），写进 `CMAKE_CL_SHOWINCLUDES_PREFIX` 的
`CMAKE_USER_MAKE_RULES_OVERRIDE` 文件再 configure。**`scripts/windows/build-backends.ps1`
（构建 ninfer 用的那个）缺这一步**，现已补上，两者行为一致；`-SkipMsvcIncludePrefix` 可关闭。

修复后 rules.ninja 里是正确的 UTF-8，实测：

| 检查 | 修复前 | 修复后 |
|---|---|---|
| `ninja -t deps <某 .cpp.obj>` | `#deps 0` | 完整列表（`program_impl.cpp.obj` 117 条） |
| 碰 `include/ninfer/types.h` 后重编的 **CXX** 对象 | **0** | **132** |
| 碰同一头后重编的 CUDA 对象 | 44 | 44 |
| `kvmem/include/kvmem/kvmem_store.hpp` 在依赖里 | 否 | 是（`memory_statistics.cpp.obj`、`kvmem_window.cpp.obj` 各含 7 个 kvmem 头） |

即：**本文上面那个崩溃的具体成因（构造 `KvMemStore` 的 TU 没重编）也随之消失。**

仍然保留的手工兜底（绕过了探测、或改过 CMake 缓存时用）：

```powershell
$bd = 'I:\llama\kvmem-llama.cpp-multi-backend-framework\build-backends\ninfer'
Get-ChildItem $bd -Recurse -Filter '*.obj' | Remove-Item -Force
Get-ChildItem "$bd\kvmem-core" -Recurse -Filter '*.lib' | Remove-Item -Force
scripts\windows\build-backends.ps1 -Backend ninfer -CudaPath <cuda> -Jobs 10
```

## 规则（重要）

1. **改任何参与编译的头文件后，先确认依赖真的被跟踪**，再相信增量构建：

   ```powershell
   ninja -C <build-dir> -t deps <某个 .cpp.obj>
   ```

   列出头文件才算正常；`#deps 0` 就是这个缺陷复发。

2. **不要手工 `cmake .` 之后直接构建**。`CMAKE_USER_MAKE_RULES_OVERRIDE` 能让前缀在重配置中存活，
   但一旦有人绕过它重新 configure，CMake 会再次写入乱码前缀，此后**新编的** C++ 对象又丢依赖。
   走 `build-backends.ps1` 的标准路径，或手工 configure 时显式带上
   `-DCMAKE_USER_MAKE_RULES_OVERRIDE=<build-dir>/kvmem-msvc-rules.cmake`。

3. 改动 `kvmem/include/kvmem/*.hpp` **类成员布局**时，即使依赖已被跟踪，也建议先全量重编一次——
   ABI 错配的代价（静默野指针写、或启动即报校验失败）远高于一次全量重编。
