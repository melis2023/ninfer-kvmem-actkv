# DFlash2 × KVMem 修复记录（2026-10-18）

> 结论先行：修复后，KVMem 缝合窗口下 DFlash2 的 2048 滑窗草稿恢复正常识别。
> 模型草稿接受率从 **0%** 恢复到 **33-55%**（greedy），256k 召回测试 decode
> 提速 **~50-87%**，召回正确性 **33/33** 分毫未损。

## 1. 现象

| 场景 | 修复前草稿接受率 | 修复前 decode |
|---|---:|---:|
| 短上下文（77/598/1118 token，greedy） | 32.6-55% | 76-79 tok/s |
| 上下文 > ~2048 token（2366/3510/4654） | **0%** | 30-31 tok/s |
| 131k 缝合窗口（greedy） | **0/36** | 71.4 tok/s（全靠 ngram） |
| 长思考（温度 1.0，HTML 规划类） | ~0% | 24.4 tok/s |

二分实验定位：阈值恰在 **2048**（DFlash2 环容量）附近；与重复文本、检索缝合、
采样温度均无关（8k 非重复文本同样 0 接受）。

## 2. 根因

### 2.1 主因：`dflash_envelopes` 的 max_context 短路（envelope 语义错误）

DFlash2 的 conditioning 环按**绝对 logical 位置**寻址（slot = `p mod 2048`），
query 位置随上下文增长（256k 场景可达 24 万+）。

而滑窗注意力内核（`sliding_window` split/reduce 两个 kernel）带防御检查：

```cpp
if (length < 0 || length > max_context || valid < 0 || valid > Tokens) return; // 输出全 0
```

`dflash_envelopes(min_frontier, max_frontier)` 把 local envelope 的 `max_context`
设成 `max_frontier`，调用点在 graph capture 时传入的是**环容量 2048**
（`graphs.cpp` 的 `draft_envelope_max = dflash_ring`）。**CUDA graph 在 capture
时把 `max_context=2048` 固化进图**——于是任何 query 绝对位置 > 2048 的草稿轮
attention 全部静默输出零，草稿 logits 变垃圾，接受率归零。运行时
`decode.cpp` 里重新计算的 envelope 只影响 host 端 plan（split 数），改不了
图里已固化的值。

这解释了全部现象（2048 阈值、与内容无关），也是 13:17 那版
logical/compact 坐标修改没能修好的原因——坐标改对了，envelope 没动。

**修复**：`dflash_envelopes` 增加第三个参数 `local_absolute_max`，
local 环 envelope 的 `max_context` 传**逻辑上限**（`capacity` = 262144），
full envelope（DFlash v1 的 Full pool，compact 空间）保持窗口语义。4 个
调用点全部更新（`decode.cpp` eager/graph 各一处、`graphs.cpp` warmup/capture
各一处）。

### 2.2 KVMem 检索缝合污染环（draft floor 掩码）

KVMem 检索重选（窗口满 B 或强制选择）会换掉窗口中部块；DFlash2 环里残留
被换出 token 的陈旧特征——草稿读到"缝合文本"，质量崩塌。环容量 = bat 的
`--kvmem-recent-tokens 2048`，所以尾部 recent 段在重选后仍保证连续。

**修复**：

- `SequenceState` 新增 `dflash_context_floor`：环内有效特征的最小 logical 位置。
- `kvmem_window.cpp` 检索重选分支与恢复路径、`checkpoint_recovery.cpp` 恢复
  路径统一在窗口重选/恢复时提升 floor 到尾部连续段起点
  （`frontier - recent_blocks*64`）。
- 内核（split + reduce）新增 per-batch `min_positions` 参数：位置低于 floor
  的环 slot 在 attention 中不可见（`context_count` 下限收紧）。公开 API
  `sliding_window_attention` 增加可空参数，null 等价禁用（DFlash v1 与
  其他调用方传空）。

### 2.3 state fork 槽位跟随（防御性）

环按 physical StateImage slot 绑定（每槽一个 lane）。prefill 多 chunk 时
checkpoint 捕获可能 fork state、切换 write 槽，特征会写进旧槽、decode 读
新槽。本轮实测未触发 fork，但代码路径真实存在（`prefill.cpp` sink 注释
明说 "a checkpoint may fork the destination slot between chunks"）。

**修复**：`SequenceState` 新增 `dflash_state_slot` 记录环最后写入的物理槽；
prefill 每 chunk 与 decode 每轮检测 `state_selectors` 的 destination 变化，
变化时用现成的 `save_rewrite_checkpoint`（`CyclicKVCache::copy_slot_from`，
此前无调用者）把环整槽复制到新槽，并打印
`KVMEM_DFLASH_FORK slot x -> y` 诊断行。checkpoint 恢复路径重新锚定槽位。

### 2.4 接受率诊断与止损（bypass，仅诊断）

`RequestControl` 新增 `draft_bypass_strikes` / `draft_bypass`：连续 6 个
"神经轮 0 接受"（ngram 复制轮不计数、接受过 1 个即清零）打印
`KVMEM_DFLASH_BYPASS` 诊断行。

注：早期版本曾把 bypass 路由到 `decode_ordinary_batch`，会因 masked-draft
后端没有 ordinary 执行资源而崩溃（"ordinary batch execution requires the
ordinary backend"），已移除路由，只保留诊断。真正实现止损需在 dflash 批内
支持 k=0 无草稿轮（见后续）。

### 2.5 dflash 轮缺失 `prepare_window`（窗口满时 entitlement 溢出）

普通轮与 MTP 轮每轮都调用 `prepare_window` 为当轮宽度准备 KVMem 窗口
（必要时触发检索重选/压缩）；dflash 轮从来没有。窗口未满时无碍，但
一旦 B+R 设备窗口填满（长会话/长生成耗尽 `--kvmem-gen-reserve`），加宽的
验证（extent+1 列）会请求超过 entitlement 的覆盖：

```
ERROR engine | worker crash: KV coverage exceeds active entitlement:
tokens=53250 required_pages=833 mapped_pages=832 reserved_pages=0 entitlement=832
```

53250 = 53248（B+R）+ 2，正是"满窗口 + 一轮 3 草稿验证"的溢出量。这个
bug 在修复 2.1 之前被掩盖（草稿轮静默输出零时请求宽度仍相同，但窗口通常
没被真正塞满）；接受率恢复后长生成立刻暴露。

**修复**：`decode_dflash_batch` 的逐行循环里补上
`prepare_window(sequence, frontier, min(capacity, frontier + width))`，
`width = verify_drafts + 1`，与 MTP 轮一致。重选路径同时会提升
`dflash_context_floor`（2.2 的补丁），环掩码语义保持一致。

## 3. 改动文件

| 文件 | 改动 |
|---|---|
| `src/models/qwen3_5/program/context_work.{h,cpp}` | `dflash_envelopes` 增加 `local_absolute_max` 参数 |
| `src/models/qwen3_5/program/decode.cpp` | envelope 调用点、ingress `draft_floors` 填充、slot fork 检测、bypass 统计、**dflash 轮 `prepare_window`** |
| `src/models/qwen3_5/program/graphs.cpp` | graph capture envelope 调用点 |
| `src/models/qwen3_5/program/program_impl.h` | `SequenceState` + `dflash_context_floor` / `dflash_state_slot`；`RequestControl` + bypass 字段 |
| `src/models/qwen3_5/program/prefill.cpp` | chunk 循环 slot fork 检测 + 环复制 |
| `src/models/qwen3_5/program/storage/kvmem_window.cpp` | 重选/恢复时提升 draft floor、重新锚定槽位 |
| `src/models/qwen3_5/program/planning/checkpoint_recovery.cpp` | 恢复路径 floor + 槽位锚定 |
| `src/models/qwen3_5/program/round_buffers.{h,cpp}` | `DFlashDecodeIngress`/`DFlashDecodeState` + `draft_floors` |
| `src/models/qwen3_5/execution/draft.cpp` | DFlash2 滑窗注意力调用传 floor（DFlash v1 传空） |
| `src/ops/softmax_attention/sliding_window/`（h/cpp/launch/kernel） | `min_positions` 掩码参数（split + reduce 内核） |
| `include/ninfer/ops/sliding_window_attention.h` | 公开 API 签名与文档 |

## 4. 验证

### 4.1 二分对照（greedy，修复后）

| prompt token | 草稿接受率 | decode |
|---|---:|---:|
| 598 | 44.4% | 79.4 tok/s |
| 1,118 | 47.7% | 78.0 tok/s |
| 2,366 | **43.2%**（修复前 0.4%） | **72.7 tok/s**（修复前 30.9） |
| 3,510 | **39.6%**（修复前 1.5%） | **72.8 tok/s**（修复前 31.3） |
| 4,654 | **41.8%**（修复前 1.5%） | **75.7 tok/s**（修复前 31.8） |

### 4.2 131k 缝合窗口（greedy）

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 模型草稿 | 0/36 (0%) | **6/18 (33%)** |
| 总接受（含 ngram） | 29/67 (43.3%) | **35/49 (71.4%)** |
| decode | 71.4 tok/s | **120.3 tok/s（+68%）** |
| prefill | 404.9 tok/s | 420.2 tok/s |

### 4.3 256k 召回（niah33_262k.py，bat 参数服务器）

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 同轮探针 | 33/33 | **33/33** |
| 终轮召回（strict） | 33/33 | **33/33** |
| 终轮召回（union） | 33/33 | **33/33** |
| 跨工具复验 | 14/14 | **14/14** |
| 写入 prefill | 394-939 tok/s | 387-961 tok/s（持平） |
| 探针 decode | ~30-40 tok/s | **58.7-98.5 tok/s** |
| 终轮召回 decode | ~72.6 tok/s | **106.0-135.8 tok/s（+46%~+87%）** |
| 总耗时 | ~12 min | 11.8 min |

全程无 ERROR/FATAL、无 `KVMEM_DFLASH_BYPASS` 触发（草稿在赚钱）、无 fork 异常。

### 4.4 长生成压力（B/R=1024 小窗口，6000 token 输出）

修复前该场景在窗口塞满后必然触发 `tokens=53250` 式崩溃；修复后：

| 指标 | 结果 |
|---|---|
| 崩溃 | **无**（6000 token 完整生成，`finish_reason: length`） |
| dflash2 接受率 | **3,233/8,215 (39.4%)**（窗口反复重选 6 次下草稿依然工作） |
| decode | 59.8 tok/s |

## 5. 后续建议

1. **bypass 的 k=0 无草稿轮**：在 dflash 批内支持 `verify_drafts=0`、
   `proposal_k=0`（跳过草稿、宽度 1 验证、eager 执行），让"内容型不匹配"
   场景（如 HTML 规划类思考，温度 1.0 下持续 0 接受）也能止损提速。当前
   bypass 仅打日志。
2. **采样温度**：greedy 下草稿接受率 33-55%，`temperature 1.0` 下天然减半。
   想最大化 DFlash2 红利，思考阶段建议低温度 / `top_k 1`。
3. **`--kvmem-recent-tokens` 与环容量**：floor 方案依赖
   `recent_tokens >= 2048`（= 环容量）才能零上下文损失；若调小 recent，
   每次重选会牺牲一部分草稿上下文（可接受但值得知晓）。
