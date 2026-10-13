# ActKV §5.2 GQA-max + §6 置信度预算接线 实现记录

| 项目 | 内容 |
|---|---|
| 论文 | ActKV: Efficient LLM Agents through Action-Guided KV Cache Management（arXiv:2609.31395v1） |
| 目标 | `I:\llama\kvmem-llama.cpp-multi-backend-framework` |
| 后端 | **ninfer**（`backends/ninfer`，已打补丁、可构建）；llama.cpp 后端本轮未动 |
| 日期 | 2026-10-11 起，2026-10-12 完成 A/B |
| 最终结论 | **§5.2 GQA max：实测回退，已回滚为 `mean`**；**§6 置信度预算：已接线、实测可用、默认关闭** |

---

## 1. 改了什么

### 1.1 §5.2 GQA max（已回滚）

`workers/ninfer/start-kvmem-ninfer.bat` 曾被我改为 `--kvmem-score-agg max`，
**A/B 实测证明是回退（见 §5），已改回 `mean`**。

代码链路本来就是通的（`serve_options.cpp` 注册 → `kvmem_window.cpp` 传
`MeanKeyGroupReduce::Max` → 共享库 `reduce_max` 分支），缺的只是启动脚本没传。
**注意**：此前脚本注释第 42-46 行一直在宣传 `max`，但实际命令行没传 —— 这个"注释与实现不一致"
已在本轮修正（注释现在写明 A/B 结论）。

### 1.2 §6 置信度自适应预算（从零接线，保留）

论文 §6.1/§6.2 三段（token 置信度 → min-pool/bottom-b%/线性拟合 → 斜率驱动扩容）
原本只有 `KvMemConfidenceMonitor` 一个类，**生产零调用**。本轮补齐：

| 层 | 落点 |
|---|---|
| 参数 | `backends/ninfer/include/ninfer/types.h` 新增 6 个 `conf_*` 字段 |
| 参数 | `backends/ninfer/src/serve/serve_options.cpp` 新增 6 个 `--kvmem-conf-*` + `--help` + 校验 |
| 核心 | `kvmem/include/kvmem/confidence_monitor.hpp`、`kvmem/src/host/confidence_monitor.cpp` 新增 `token_confidence_from_logprobs()` 与 **`KvMemConfidenceBudget`** |
| 采集 | `decode.cpp`（ordinary / MTP / DFlash 三条路径）+ `prefill.cpp` 的 `observe_confidence()` |
| 门控 | `kvmem_window.cpp` `prepare_window()`：**每个压缩轮**调 `advance_round(begin)` |
| 创建 | `memory_statistics.cpp`：仅当 `--kvmem-conf-budget` 非 0 时创建 |

---

## 2. 三个关键设计决策（论文没写）

### 2.1 `T_i` 不重算词表，直接复用引擎已有的 top-k logprob

ninfer 的 `logprob_topk` Op 已输出 `top_values[k] = s[v] - lse`，**这就是 `log P(v)`**，
且 `kLogprobTopK = 20` 与论文 §6.1 的 `k=20` 完全一致。所以
`token_confidence_from_logprobs()` 只需取负平均，不需要再做全词表 reduce。

### 2.2 预算增长只在**压缩轮**发生，并加冷却 + 幂等（最关键）

论文说"每个压缩步检查一次"，但 `should_increase_budget()` 是 pooled ring 的**持续属性**，
不是一次性事件。**按 token 调用会 4096 → 7168 → 12544 → 21952 → 36864，4~6 个 token 就顶到上限**，
预算退化成二值，省内存的意义完全消失。

`KvMemConfidenceBudget` 强制四条：

1. `advance_round()` 只在压缩边界调用（`prepare_window`），**绝不 per token**；
2. `cooldown_rounds`（默认 4）内至多增长一次；
3. 增长后 `reset()` 监控窗口，避免陈旧的负斜率立刻触发下一次；
4. `round_key`（传 `begin`）幂等 —— `prepare_window` 的异步路径会 `publish()` 后递归重入，
   同一轮不能算两次。

### 2.3 `B_upper = selected_tokens`，因此**不动物理布局**

ninfer 的设备 KV 池在启动时按 `B + R` 固定分配，并由 `model_instance.cpp:328-331` 强制
`kv_capacity == (B+R) * lanes`。所以 `upper_tokens = kvmem_options.selected_tokens`，
并在 `serve_options.cpp` 增加校验（64 对齐 / ≤`--kvmem-budget` / ≥`sink+recent`）。

> **如实说明的限制**：该功能**换不到显存**。`pick_topk_blocks` 的窗口会变小，
> 但 `MemoryBudget.selected_tokens`（plan/校验用）仍是固定的 `--kvmem-budget`，
> 物理池也按 B+R 分配。收益是"降低平均注意力窗口长度"，不是"降低显存占用"。
> 要真正做到"起步只分配小池"，需改 `model_instance.cpp` 的池分配器，属独立改动，**本轮未做**。

---

## 3. 验证证据

| 检查 | 结果 |
|---|---|
| 宿主单测 `ctest`（全量） | **15/15 通过** |
| 其中 `confidence_monitor_test` | **通过**（本轮新增 10 个用例） |
| ninfer `ninfer-serve` 全量构建 | **成功**（Release + CUDA sm120a） |
| `--help` 列出 6 个新参数 | ✅ |
| `prepare-backends.py --backend ninfer --check` | **ready** |
| 补丁反向应用 `git apply --reverse --check` | **rc=0** |
| `versions.json` 的 `patch_sha256` | **一致** |

**顺带修掉一个既有缺口**：`confidence_monitor_test.cpp` 文件存在但**从未注册进
`kvmem/CMakeLists.txt`**，此前从未被编译或运行 —— 文档里"已单测通过"的说法是空的。
本轮已注册。

新增单测（10 个）：`test_logprobs_match_token_confidence`、`test_logprobs_rejects_bad_input`、
`test_budget_grows_only_on_negative_slope`、**`test_budget_is_rate_limited_per_round`**（回归）、
`test_budget_never_exceeds_upper`、`test_budget_never_shrinks`、`test_budget_disabled_when_start_zero`、
`test_budget_rounds_to_granularity`、`test_budget_ignores_nan`、`test_budget_round_key_is_idempotent`。

---

## 4. 端到端实测（2026-10-12）

| 测试 | 结果 |
|---|---|
| 服务启动（完整 launcher 参数） | ✅ `listening on http://127.0.0.1:18200` |
| 短请求推理 | ✅ `回复 [BUILD OK]` |
| §6 开启（`--kvmem-conf-budget 8192`） | ✅ 服务启动 |
| 8,059-token 长上下文（触发压缩路径） | ✅ `回复 [LONG OK]`，prefill 1.08k tok/s |
| 6 条参数校验规则 | ✅ 全部正确触发 |
| 服务端异常 | ✅ 无 error/exception |

校验实测输出：

```
--kvmem-conf-budget 100      -> must be a positive multiple of 64 tokens
--kvmem-conf-budget 40960    -> must not exceed --kvmem-budget (B_upper)
--kvmem-conf-scale-up 0.5    -> must be > 1
--kvmem-conf-bottom 2.0      -> must be in (0, 1]
--kvmem-conf-pool 0          -> must be > 0
--kvmem-score-agg bogus      -> must be mean or max
```

---

## 5. 关键实测结果：GQA max 是**回退**，已回滚

### 5.1 A/B（唯一变量 `--kvmem-score-agg`，40 针 / 256k，同二进制、同机器）

| 配置 | 同轮召回 | 最终召回 | 重试后并集 | 工具调用 |
|---|---|---|---|---|
| **`mean`（保留）** | **39/40** | **39/40** | **39/40** | 80/81 |
| `max`（论文 Alg.3 step 8） | 28/40 | 32/40 | 33/40 | 68 |

逐轮同轮召回：

```
mean: 4/4 3/3 3/3 3/3 3/3 3/3 3/3 3/3 2/3 3/3 3/3 3/3 3/3
max : 4/4 3/3 3/3 3/3 3/3 2/3 2/3 1/3 1/3 2/3 2/3 1/3 1/3
                              ^ 第 6 轮起渐进劣化
```

### 5.2 为什么论文的 max 在这里反而更差

劣化是**渐进式**的（3/3 → 2/3 → 1/3），不是随机丢针，说明是**累积效应**：

1. `max` 取 GQA 组内**单个最强 head**，`mean` 是组内折叠平均。某个 head 偶然峰值就会主导
   整块的 `AttnScore`，使分数**方差更大**；
2. 这个更噪的分被喂进 `KS_t = λ·KS_{t-1} + hit·score` 的 LRFU 累积。λ=0.5 的记忆效应会把
   噪声**逐轮放大**，排名逐渐偏离"跨轮稳定的 action-critical 集合"；
3. 结果：action-critical 块在后续轮次**逐渐丢掉槽位**，表现为越往后越差。

这印证了此前把 `mean` 作为默认是对的（见
[kvmem-actkv-optimization-notes.md](kvmem-actkv-optimization-notes.md) 的 A4 条），
当时只给了"6 倍开销"的性能理由，现在有了召回率证据。

### 5.3 复现

```powershell
# 服务（脚本已回 mean）
I:\llama\kvmem-llama.cpp-multi-backend-framework\workers\ninfer\start-kvmem-ninfer.bat
# 测试（约 12 分钟）
python I:\dsh\html\niah40_256k.py
```

产物：`I:\dsh\html\AB-mean.log`（39/40）、`I:\dsh\html\AB-max.log`（28/40 同轮）。

A/B 期间**未启用 §6**，所以上表是纯粹的 score-agg 对比。

---

## 6. 未验证项

1. **§6 未做召回率 A/B**。只验证了"能启动、能生成、校验生效"，未测开启后召回率是否变化。
2. **§6 的收益场景存疑**。单会话打满时池固定为 B+R，**没有显存收益**；且你的场景第 4 轮就填满池，
   小预算只影响前 3 轮。建议**不启用**，除非跑多会话波动负载。
3. **§6 缺少可观测性**：没有打印每轮 `rounds/growths/tokens` 的 trace，
   无法知道预算实际走了几步、停在哪。**建议补 `KVMEM_CONF_TRACE`。**
4. **llama.cpp 后端未同步**：`backends/llamacpp` 是空子模块、根 `llama.cpp/` 是未打补丁上游，
   该后端当前编不出来；共享库改动两边共用，但 §6 接线是 ninfer 专有。

---

## 7. 构建环境的严重缺陷（务必知悉）

**本机 `build-backends/ninfer` 的 C++ 头文件依赖完全丢失**：

```
CMakeFiles/rules.ninja:
  msvc_deps_prefix = 娉ㄦ剰: 鍖呭惈鏂囦欢:      <-- 应为 "注意: 包含文件:"
```

原始字节是 `注意` 被**双重编码错误**的产物（GBK 误解码后又存成 UTF-8）。
ninja 拿这个乱码匹配 MSVC 的中文输出，**永远匹配不上**：

```
C++  目标文件 329 个 -> 依赖数全部为 0   <-- 头依赖完全丢失
CUDA 目标文件 405 个 -> 依赖数正常
```

**后果**：改任何 `.h` 都不触发包含它的 `.cpp` 重编，静默产生 **ABI 不一致**的二进制。
本轮就因此编出过启动即 `0xc0000005` 的坏二进制（`KvmemOptions` 加字段后
`main.cpp.obj` 未重建，按旧布局读 `ServeOptions`）。

**可靠做法（改任何头文件后必须执行）**：

```powershell
Get-ChildItem backends\ninfer -Recurse -Include *.cpp,*.h,*.cu,*.cuh -File |
  Where-Object { $_.FullName -notmatch '\\third_party\\' } |
  ForEach-Object { $_.LastWriteTime = Get-Date }
```

注意必须包含 **`apps/`**（第一次我漏了它，`main.cpp.obj` 一直是旧的）。
另外：用 `Copy-Item` 恢复源文件会**保留旧时间戳**，ninja 会误判为无需重建。

**根治**：删掉 `build-backends/ninfer` 让 CMake 用正确 locale 重新生成 `rules.ninja`
（`VSLANG=1033` 实测**无效**）。需全量重编，尚未执行。

---

## 8. 构建命令（本机已验证）

```powershell
# 需要 VS 环境 + vcpkg 环境变量（否则 FFMPEG 找不到）
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
  -version '[17.0,18.0)' -latest -products * `
  -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
& "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$env:VCPKG_ROOT='I:\vcpkg'; $env:VCPKG_TARGET_TRIPLET='x64-windows'

# 宿主测试
cmake -S . -B build-kvmem-host -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF
cmake --build build-kvmem-host --parallel 8
ctest --test-dir build-kvmem-host

# ninfer
cmake -S backends/ninfer -B build-backends/ninfer -G Ninja -DCMAKE_BUILD_TYPE=Release `
  "-DNINFER_KVMEM_SOURCE_DIR=$PWD/kvmem" -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=OFF `
  -DCMAKE_CUDA_ARCHITECTURES=120a -DNINFER_SM120_NATIVE=ON
cmake --build build-backends/ninfer --parallel 8 --target ninfer-serve
```

> **坑**：`build-backends/ninfer` 的 CMake 缓存若缺少 `VCPKG_ROOT`/`VCPKG_TARGET_TRIPLET`，
> 重新配置会因 `FindFFMPEG.cmake` 找不到而失败。这两个变量必须在 configure **之前**设置。

---

## 9. 回滚

**运行时**（推荐，最简）：去掉 `--kvmem-conf-budget` 即可，默认关闭、行为与改动前一致。
`--kvmem-score-agg` 已是 `mean`（与基线相同）。

**代码**：

```powershell
git -C I:\llama\kvmem-llama.cpp-multi-backend-framework checkout -- `
  backends/patches/ninfer-kvmem.patch backends/versions.json `
  kvmem/CMakeLists.txt kvmem/include/kvmem/confidence_monitor.hpp `
  kvmem/src/host/confidence_monitor.cpp kvmem/tests/confidence_monitor_test.cpp `
  workers/ninfer/start-kvmem-ninfer.bat
git -C I:\llama\kvmem-llama.cpp-multi-backend-framework/backends/ninfer checkout -- `
  include/ninfer/types.h src/models/qwen3_5/program/decode.cpp `
  src/models/qwen3_5/program/prefill.cpp src/models/qwen3_5/program/program_impl.h `
  src/models/qwen3_5/program/storage/kvmem_window.cpp `
  src/models/qwen3_5/program/storage/memory_statistics.cpp src/serve/serve_options.cpp
python scripts\prepare-backends.py --backend ninfer
```

> **警告**：`backends/patches/ninfer-kvmem.patch` 是基于**完整工作树**（含仓库原有未提交改动，
> 例如 `InvalidToolConstraint` 枚举扩展）生成的。直接 `checkout` 会让补丁与工作树不一致，
> 需要重新生成并更新 `versions.json`。
