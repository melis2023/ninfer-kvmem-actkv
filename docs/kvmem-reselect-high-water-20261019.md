# KVMem 重选水位提前到 80%（2026-10-19）

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
