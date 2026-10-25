# KVMem 重选水位提前到 80%（2026-10-19）

> **2026-10-26 已回退**：80% 水位在 `B > 4R` 时每个 chunk 都空转重选（issue #6），
> 密集重选还会把 `removed_tokens` 推到 checkpoint 前沿之前、使 253K prefill 崩溃。
> 现在任何 B/R 都回到 `compact_position > W` 的原始 100% 触发，见第 11 节；
> 第 1–10 节保留为当时的数据记录。

## 1. 动机

窗口 `W = B + R = 36,864 + 16,384 = 53,248` token，原来在 `compact_position > W`
**100% 满**时才触发重选。用户提出两个想法：

1. 在 80% 水位就开始交换，给传输留出余量；
2. decode 时边驱逐、边召回、边计算。

第 2 点需要双缓冲窗口（额外 1.68 GiB 显存），当前 16 GiB 卡在
模型 11.3 GiB + 运行时后只剩 308 MiB，不可行。本次只落地第 1 点。

## 2. 改动

`backends/ninfer/src/models/qwen3_5/program/storage/kvmem_window.cpp:633`

```cpp
// 原来
const bool reselecting =
    force_selection || compact_position(sequence, end) > kvmem_window_tokens;

// 现在
const std::uint32_t reselect_high_water =
    kvmem_window_tokens - kvmem_window_tokens / 5;   // 42,598
const bool reselecting =
    force_selection || compact_position(sequence, end) > reselect_high_water;
```

选块算法、mandatory 集合、预算都没有改，只把触发点从 100% 挪到 80%。

## 3. 机制：快的原因不是“重选更频繁”

重选后视图压缩到 `selected_tokens = 36,864`，R 从 0 重新增长。

| 水位 | 重选触发点 | 重选后窗口 | 平均窗口 |
|---|---:|---:|---:|
| 100% | 53,248 | 36,864 | 45,056 |
| 80% | 42,598 | 36,864 | **39,700** |

平均窗口缩小 12%，attention 计算量正比于窗口，所以：

- prefill 受益最直接（每 token 都要扫窗口）
- decode 受权重读取主导，收益较小，但同时接受率提升

## 4. 64K 实测

payload：`workers/ninfer/sweep.json`（59,336 prompt，max_tokens 200）
配置：MTP `--draft-tokens 3`，`--kvmem-sink-tokens 1024`

| 指标 | 100% 水位 | 80% 水位（两次） | 变化 |
|---|---:|---:|---:|
| prefill | 419.18 | 434.07 / 411.26（均值 422.67） | +0.8% |
| decode | 45.40 | 46.71 / 46.11（均值 46.41） | **+2.2%** |
| 每 token | 22.02 ms | 21.41 / 21.69（均值 21.55） | −2.1% |
| 草稿接受率 | 42.5% (111/261) | 51.3% (120/234) | +8.8 pp |
| 重选次数 | 2 | 7 | 3.5× |

## 5. 256K 实测

payload：`workers/ninfer/sweep256k.json`（247,064 prompt，max_tokens 200）

| 指标 | 100% 水位 | 80% 水位 | 变化 |
|---|---:|---:|---:|
| 端到端 | 724.2 s | **637.1 s** | **−12.0%** |
| prefill | 343.4 | **390.5** | **+13.7%** |
| decode | 45.4 | **49.5** | **+9.0%** |
| 每 token | 22.0 ms | 20.2 ms | −8.2% |
| 草稿接受率 | 42.5% | 47.3% (116/245) | +4.8 pp |
| 重选次数 | 24 | 75 | 3.1× |
| 传输总计 | 6.14 s | 17.51 s | +185% |
| 分数计算总计 | ~1 s | ~12.8 s | 12× |

### 时间账

```
纯 attention 计算   −110 s   （窗口小 12%）
重选开销            + 23 s   （传输 +11.4 s，打分 +11.8 s）
------------------------------------------
净收益              − 87 s
```

## 6. 召回验证：33/33

`python niah33_262k.py`，15 轮 × 435 filler 行，33 针，80 次工具调用，
全程 756.9 s，92 次重选：

```
same-turn recall (fresh needles):    33/33
final recall (strict, first pass):   33/33
final recall (after 1 retry, union): 33/33
cross-tool (send_alert) recall:      14/14
```

与 100% 水位的 33/33 一致，窗口缩小 12% 对这套测试无影响。

### 阶段速度

| 阶段 | tok/s |
|---|---:|
| write prefill | 351–985 |
| probe prefill | 105–194 |
| probe decode | 52–84 |
| recall prefill | 335–350 |
| recall decode | 90–113 |

## 7. 代价与风险

| 项 | 变化 | 说明 |
|---|---|---|
| 平均可见上下文 | −12% | 45,056 → 39,700 token |
| 重选次数 | 3.1× | 每次重选间隔从 16,384 降到 5,734 token |
| 传输总量 | +185% | restore 每次仍是全量 576 页 |
| 打分总量 | 12× | 每次重选扫全部历史页 |
| 召回 | 无变化 | 33/33 |
| 首 token 延迟 | 略降 | prefill 更快 |

最大风险是“多细节精确回忆”场景，33 针测试覆盖不到同时引用 10+ 个远距离事实。
如果出现退化，可把水位回调到 90%（`kvmem_window_tokens / 10`），窗口平均
42,500，收益约 6%。

## 8. 下一步：差异 restore

日志显示每次重选：

```
KVMEM_RESTORE pages=576    ← 全量恢复
KVMEM_SPILL   pages=88     ← 只写回差异
```

75 次 × 576 页 = 43,200 页，实际每次只有 88 页是新的。如果 restore 也走差异：

| 项 | 现在 | 差异 restore |
|---|---:|---:|
| 每次 restore | 576 页 | 88 页 |
| 75 次总 restore | 43,200 页 | 6,600 页 |
| 传输总时间 | 17.5 s | ~2.6 s |

这是零风险的纯优化，且对 100% 水位同样有效。

## 9. 回退

```powershell
# 恢复 100% 水位
git -C backends/ninfer checkout -- src/models/qwen3_5/program/storage/kvmem_window.cpp
# 或把 high_water 改回 kvmem_window_tokens
```

## 10. 复现命令

```powershell
# 服务
ninfer-kvmem-server.exe `
  I:\model\ninfer\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.ninfer `
  --port 18200 --model-id ninfer-27b --max-context 262144 `
  --kvmem-sink-tokens 1024 --prefill-chunk 256 `
  --kvmem-budget 36864 --kvmem-gen-reserve 16384 --kvmem-host-mib 12288 `
  --kvmem-lrfu --kv-dtype int8 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 `
  --spec mtp --draft-tokens 3 --ngram-draft-tokens 31 --adaptive-mtp --full-draft-head `
  --vision --vision-residency cpu --vision-max-merged 256 `
  --default-reasoning-effort xhigh --default-thinking-budget 16384

# 64K / 256K
curl.exe -s -X POST http://127.0.0.1:18200/v1/chat/completions `
  -H "Content-Type: application/json" --data-binary "@workers/ninfer/sweep.json"
curl.exe -s -X POST http://127.0.0.1:18200/v1/chat/completions `
  -H "Content-Type: application/json" --data-binary "@workers/ninfer/sweep256k.json"

# 召回
python niah33_262k.py
```

## 11. 2026-10-26 结论：放弃 80%，回到 100%

### 触发条件

80% 水位是 `W - W/5`，重选后视图压到 `B = selected_tokens`。当 `B > W - W/5` 时，
压缩结果本身就落在水位之上：

```text
B > W - W/5,  W = B + R
=> B > 4 / 5 (B + R)
=> B > 4R
```

于是每个 prefill chunk（256 token）都会触发一次重选：一次 200 ms 级的全窗口 restore
只换来 256 token 的前进。issue #6 报的就是这个。

### 试过但不采用的三种修法

| 方案 | 做法 | 结果 |
|---|---|---|
| 增长门限 v1 | `last_reselect_visible` + `R/2`，基线取 `compact_position` | 253K prefill 崩溃 |
| 增长门限 v2 | 基线改成 `selected_tokens` / `0` | 同样崩溃 |
| 条件回退 | `B > 4R` 用 100%，否则 80% | 能跑，但要维护两套触发点 |

两版增长门限的崩溃点相同：

```text
ERROR engine | worker crash: position precedes compact KV append frontier
```

密集重选把 `removed_tokens` 推高，checkpoint restore 回到旧前缀时
`prepare_window(frontier, frontier, true)` 里的 `compact_position(frontier)` 落在
append frontier 之前。换基线不解决，因为问题出在重选频率本身。

条件回退版在 256K NIAH 上 3/3、也不再崩溃，但保留两套触发点没有收益：
80% 的收益（10-19 记录的 256K 端到端 −12%）是真实的，代价却是每次改 KVMem 都要
同时验证两条路径，而且 `B > 4R` 永远走不到 80%。

### 最终代码

```cpp
if (end == begin && !force_selection) { return true; }
const bool reselecting =
    force_selection || compact_position(sequence, end) > kvmem_window_tokens;
```

- 任何 B/R 都只在窗口满（`W = B + R`）时重选；
- 两次重选之间至少增长一个 R，不会出现 no-op；
- 不新增状态，不碰 pending / checkpoint 路径；
- 启动时不再有 `B > 4R` 的 WARN（它只为 80% 水位存在）。

### 验证

**会话（B/R = 55,296/13,312 = 4.15，61K prompt，1 轮）**

| 版本 | reselect | wall | 回答 |
|---|---:|---:|---|
| 80% 水位（旧二进制） | 55 | 82.2 s | `391` |
| 100%（本版） | **0** | **75.8 s** | `391` |

61K prompt 未到 `W = 68,608`，本版不触发任何重选；旧版从 54,784 开始每 chunk 一次。
启动 `probe measured 117.3 MiB`、`free 587.7 MiB`，无 WARN、无报错。

**5 轮会话（B/R=4.15，68,186 → 102,976）**

| 版本 | reselect | 5 轮总 wall |
|---|---:|---:|
| 80% 水位（旧二进制） | 759 | 634.2 s |

100% 版在同配置下的重选数与 `(输入 − W) / R` 同阶（每轮 0–2 次）。
这组 5 轮对比在回退后没有重跑。

**256K NIAH（条件回退版，B/R=4.15）**

| 项 | 值 |
|---|---:|
| 命中 | **3/3**（BLUEBIRD-42 / MAPLE-91 / REDWOOD-77） |
| reselect | 29 次（两遍 prefill 各 14 次 + 1 次 restore） |
| 触发点 | 68,608 = W，之后每次 +13,312 = R |
| prefill（报告值） | 404.2 tok/s |
| wall | 627.5 s |

该次运行在 `B > 4R` 下走的就是 100% 这段代码，所以 3/3 的召回结论对最终版同样成立：
5% 和 50% 两个 needle 都在 68,608 窗口之外，仍靠 KVMem 召回，没有为了少重选牺牲窗口。

> 253K prompt 的 prefill 实际跑两遍（一遍计入 token，一遍 hidden replay），报告值约为
> 瞬时速度的一半；370.5 tok/s 那次运行也有同样现象，与本改动无关。

### 行为矩阵

| 场景 | 80%（10-19 之后） | 100%（本版） |
|---|---|---|
| B ≤ 4R，长请求 | 80% W 触发 | 满 W 触发 |
| B > 4R | 每 chunk 一次 no-op 重选 | 满 W 触发 |
| checkpoint restore | 可能撞 append frontier | 同 80% 之前 |
| 额外状态 | `last_reselect_visible`（已删） | 无 |

### 代价

- 平均可见窗口回到 `(W + B) / 2`，比 80% 水位大约 12%（B/R=2.23 时）；
- 10-19 记录的 256K 端到端 −12% 不再有；
- 换来单一触发点、`B > 4R` 不 thrash、不会踩 checkpoint 前沿。

