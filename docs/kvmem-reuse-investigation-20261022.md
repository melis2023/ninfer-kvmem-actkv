# KV 复用「不搬页」调查（2026-10-22）

## 0. 结论

**没有可落地的召回提升。速度上的 1.4–2.5% 来自撤销测试参数，不是新优化。**

| 档位 | 召回基线 | 本轮 | 耗时基线 | 本轮 | 变化 |
|---|---:|---:|---:|---:|---:|
| LongMemEval 128K | 20/20 | **20/20** | 259.3 s | 255.7 s | −1.4% |
| LongMemEval 192K | 18/20 | **18/20** | 378.5 s | 369.2 s | −2.5% |
| LongMemEval 240K | 16/20 | **16/20** | 444.7 s | 436.0 s | −2.0% |
| NIAH 256K | 33/33, 14/14 | **33/33, 14/14** | 595.8 s | 582.3 s | −2.3% |
| 64K decode | 46.7 | **47.15** | — | — | +1.0% |

召回零损失，速度小幅提升。但第 1 节会说明：**这部分提升是把测试期间自己加的参数撤掉，生产配置本来就是快的那个。**

## 1. 方法：先定位「搬页」到底占多少

从 `KVMEM_RESTORE` / `KVMEM_RESTORE_DIFF` 全量统计（一份 64K + 256K NIAH + 三档 LongMemEval 的日志）：

| 项 | 数值 |
|---|---:|
| restore 总次数 | 449 |
| 全量（576 页） | 434 次（97%） |
| read 合计 | 30.88 s |
| sync 合计 | 41.53 s |
| **restore 总耗时** | **72.4 s** |

对照此前测过的其他项：

| 项 | 耗时 | 占 604 s |
|---|---:|---:|
| host 打分 | 10.07 s | 1.7% |
| **restore 搬页** | **72.4 s** | **12.0%** |
| prefill 计算 | ~520 s | ~86% |

搬页是打分的 7.2 倍，但 prefill 才是绝对主体。

## 2. 假设与实验

### 2.1 假设 A：增量搬页没生效

`start_restore` 的增量条件（`kvmem_window.cpp:366`）：

```cpp
const bool incremental = !checked && enough &&
    changed.size() * 2 <= selected.size() &&
    selected.size() >= std::max<std::size_t>(1, selection_pages / 2);
```

`checked` 在 `verify_restore && kvmem_options.verify_transfers` 时非空（line 210–218）。也就是说：

**只要开 `--kvmem-verify-transfers`，增量搬页永远不触发。**

实测验证：开该参数时 `inc=1` 只有 4/449 次；关闭后增量立刻生效：

```
changed=124 → pages=124, read 14ms + sync 20ms = 38ms
changed=560 → pages=576, read 65ms + sync 93ms = 158ms
```

**关键事实**：生产启动脚本 `workers/ninfer/start-kvmem-ninfer.bat` 本来就没有 `--kvmem-verify-transfers`。之前所有「基线」测试手动加了这个参数，所以基线本身带着额外开销。本轮去掉后恢复默认行为，这就是 1.4–2.5% 的全部来源。

### 2.2 假设 B：提高 LRFU 惯性可以让选择更稳

`--kvmem-lrfu-lambda` 从 0.5 提到 0.7，64K 的 changed 分布：

```
λ=0.5: 560, 560, 560, 124, 164, 124, 181
λ=0.7: 560, 560, 560, 124, 164, 124, 181
```

**完全一致。**

原因：选择变化不是历史衰减造成的，而是当前 query 的 mean-key 分数本身就变了。64K 的 4 次重选里，前两次 query 尚未完整（query 在 55K 之后，重选发生在 42.6K/48K），只能用 recent/mandatory 块；后两次 query 完整后整片重排。

### 2.3 假设 C：保留高分块，只换掉低分块

NIAH 的 `prev>0` 共 100 次，其中：

| changed 区间 | 次数 |
|---|---:|
| ≤100 | 0 |
| 101–288 | 4 |
| 289–450 | 5（含 450） |
| **>450** | **90** |

changed 均值 **505 / 576 = 88%**。

也就是说：**「保留 + 换新」的理论上限只有 12%**。按 72.4 s 算，最多省 8.6 s（1.5%）。

加上 43.8% 的 `prev=0`（跨请求页表被清空，选择与新内容无重叠），实际可省更少。

**结论：不值得为 1.5% 改动存储层。**

## 3. 为什么选择集合必然大幅变化

三个结构性原因：

1. **query 是逐步可用的。** 重选发生在 prefill 推进过程中，早期重选时 query 窗口还没 prefill 到，`query_tokens=0`，只能靠 recent/mandatory；晚期 query 完整后重选，选择整片重排。这是「用不完整输入先选一遍」的重复劳动。
2. **mean-key 是低通滤波。** 每轮 query 不同，块均值余弦的排序就不同，且 top-576 的边界附近分数密集，轻微扰动就换块。
3. **LRFU 的累积分数被当轮分数主导。** 公式 `next = λ·prev + (hit ? score : 0)` 里 hit 加的是原始 score，量级远大于衰减后的历史项，所以历史排名迁移不过去。

## 4. 已有但未使用的机制

引擎里有完整的 prefix fork：

```cpp
// kv_store.h:1064
prepare_prefix_fork(source, destination, frontier)
// kv_store.h:1177
commit_prefix_fork(fork, streams)
```

`request_plan.cpp:1062` 在 `PrivateSourceMode::Retain` 时已启用它。但 KVMem 走的是「按 query 重选块集合」路径，不是「前缀复用」，所以用不上。

**要让 KVMem 用上，需要块级物理页共享（乱序映射），改动约 300–600 行，收益上限 1.5%。当前判定不划算。**

## 5. 最终配置

```
去掉 --kvmem-verify-transfers（生产脚本本来就没有）
--kvmem-lrfu-lambda 0.5（默认）
--kvmem-lrfu-top-frac 0.9（默认）
```

**没有代码改动。** 工作区干净，子模块 `e6781461`。

## 6. 后续真正值得做的两件事

按预期收益排序：

| 方向 | 预期 | 说明 |
|---|---|---|
| 减少早期重选 | 256K 约 20 s（3%） | query 不完整时不做全局重排，只追加 recent；需验证召回 |
| 异步 restore | 约 30 s（5%） | `async=1` 当前从不出现（`memory_io` 为空）；需要接通异步 IO |

两项都不改选择算法，只改搬运时机。

## 7. 数据文件

```
I:\llama\LongMemEval\out\recall_reuseB_128000_verified.json   20/20
I:\llama\LongMemEval\out\recall_reuseB_192000_verified.json   18/20
I:\llama\LongMemEval\out\recall_reuseB_240000_verified.json   16/20
I:\dsh\html\niah33-262k-metrics.json                           33/33
C:\Users\zhuan\AppData\Local\Temp\opencode\niah-reuseB.log
C:\Users\zhuan\AppData\Local\Temp\opencode\reuseB.err.log
C:\Users\zhuan\AppData\Local\Temp\opencode\reuseC07.err.log   （λ=0.7 对照）
```

备份：`I:\llama\_backup\kvmem-llama.cpp-multi-backend-framework-20261022-021600-pre-reuse`（7.28 GB）
