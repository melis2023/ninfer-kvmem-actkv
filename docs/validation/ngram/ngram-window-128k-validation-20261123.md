# 128k 下 ngram 窗口才是瓶颈（2026-11-23）

针对「DSpine 在 128k 比历史记录慢很多」的排查结论。结论先说：**不是模式退化，也不是显存
不足触发分页，而是 `--ngram-draft-tokens` 的默认宽度太小**——128k 下每轮成本几乎固定，
窗口直接决定每轮能提交多少 token。

## 1. 测试条件

- 目标工件 `Qwen3.8-27B-GSQ-RCO-IQ3_S-q4km-mtp-dflash2.ninfer`，`--kv-dtype rk8v4`，
  `--kvmem-budget 36834 --kvmem-gen-reserve 1024`（KV 37,824 token 常驻），`--kvmem-keep`。
- 同一 prompt `C:\temp\p128k.txt`，119,123 token；输出「重复指定行 40 次」，512 token，
  temp 0，`reasoning_effort none`。所有样本 `repeats 22`（22/22 行逐字正确）。
- 每轮流程：起服务 → 第一次请求（冷 prefill + 写 cache）→ 第二次请求（100% prefix cache，
  turn closure）→ 取第二次的 decode。除特别说明，下表都是第二次请求。
- RTX 5060 Ti 16 GB，驱动侧功耗上限 180 W。实测 decode 期间 sm 2500–2870 MHz、
  功耗 140–181 W，全程贴功耗墙。

## 2. 三模式同窗口（ngram=12）：差距是噪声

同配置重复测量（每次重新加载模型），decode tok/s：

| 模式 | 样本 | 轮次 | 中位 |
|---|---|---:|---:|
| DFlash2 | 146.2 / 162.2 / 162.6 / 166.0 / 169.3 | 38 | **162.6** |
| DSpine | 141.1 / 148.6 / 152.1 / 152.3 / 158.5 / 165.4 / 172.3 / 174.3 | 37 | **155.4** |
| MTP | 137.0 / 137.5 / 155.9 / 162.0 | 38 | **146.7** |

单次样本极值 136.98 / 174.28，跨度 27%。交错 A/B（同一会话内 dflash2 → dspine → mtp
各跑一次，两轮）里排序直接翻转：

| 轮 | DFlash2 | DSpine | MTP |
|---|---:|---:|---:|
| 1 | 140.6 | **165.4** | 137.0 |
| 2 | 162.2 | **174.3** | 137.5 |

同相位的 GPU 采样（1 Hz，`gpu_cycle.csv`）显示 clock 与 decode 同步摆动：

| 相位 | decode | sm 均值 | 功耗均值 | 显存峰值 |
|---|---:|---:|---:|---:|
| c1 dflash2 | 140.6 | 2538 | 143.4 W | 15,246 MiB |
| c1 dspine | 165.4 | 2522 | 143.3 W | 15,707 MiB |
| c2 dflash2 | 162.2 | 2591 | 152.5 W | 15,210 MiB |
| c2 dspine | 174.3 | **2674** | **157.2 W** | 15,875 MiB |

即：这块卡在 128k decode 时被 180 W 功耗墙锁住，时钟与后台负载一起决定当次成绩，
单次测量 ±15% 属正常。**同窗口下三模式互差 5–8%，不足以支撑「DSpine 慢 30%」的结论。**

## 3. 排除的两个假设

**a. 显存余量不足触发 WDDM 分页？否。**

DSpine runtime 1.92 GiB，启动后 free 321.9–501.5 MiB，确实每次触发
`WARN VRAM headroom | only ... free after startup`；DFlash2 runtime 1.56 GiB，free 872.7 MiB
不触发。但：

- `--kvmem-host-mib 4096`（少 8 GiB 主机 KV）后 free 仍为 382.2 MiB → KV 在主机内存，
  不占设备预算，降预算腾不出余量；
- free 最低的那次（321.9 MiB）反而跑出冷/暖 189.4 / 158.5，是 DSpine 最好的一组成绩；
- 全程采样显存峰值 15,875 MiB / 16,311 MiB，未触顶。

free 与 decode 不相关，分页假设排除。该告警对「同一时段被别的 GPU 应用抢走几百 MiB」
仍然有效，只是不是本次差异的原因。

**b. 草稿环扫描窗口（8192 满环）太贵？否，而且这些轮次根本没有草稿前向。**

`draft.cpp` 里的 `[draft-prof]` 分解（`NINFER_DSPINE_PROFILE=1 --no-cuda-graph`）在
128k 请求里**一条都没有打印**（ngram=12 和 ngram=0 都是零输出）。原因是
`dflash_decode_batch_body`：

```cpp
if (!state.ngram || batch_size > 1) {
    propose_batch_impl(state, frame, batch_size, k, envelopes);
}
```

batch=1 且开 ngram 时神经草稿前向整段跳过，ngram 通过
`speculative_overlay_copy_proposals` 直接写入 drafts/candidates/proposal_q。所以
`NINFER_DSPINE_DRAFT_WINDOW` 的对比实验（8192/2048/1024 → 147.2/143.8/136.6）测的是噪声，
不是窗口效果——**窗口调小反而更慢，正好说明它不参与这段耗时**。

## 4. 真正的瓶颈：每轮成本固定，窗口决定每轮提交量

按 `decode = (k+1) / 每轮成本` 反算每轮耗时（512 token，取中位）：

| 配置 | 轮次 | decode | 每轮 |
|---|---:|---:|---:|
| DSpine ng12（13 列） | 37 | 155.4 | 89 ms |
| DSpine ng16（17 列） | 29 | 172.4 | 102 ms |
| DSpine ng20（21 列） | 23 | 239.7 | 93 ms |
| DSpine ng32（33 列） | 16 | 254.7 | 126 ms |
| DFlash2 ng12（13 列） | 38 | 162.6 | 83 ms |
| DFlash2 ng20（21 列） | 24 | 218.9 | 97 ms |
| MTP ng12（13 列） | 38 | 146.7 | 92 ms |
| MTP ng20（21 列） | 24 | 202.2 | 106 ms |

13 列到 21 列每轮成本基本在 83–106 ms 区间打转、并不随列数线性增长，说明每轮主导项是
**一次性权重全读**（IQ3_S 约 12.5 GiB）加常驻 KV（37,824 token × rk8v4）扫描，而不是
verify 列数；126 ms 要到 33 列才出现。既然每轮成本基本恒定，把每轮提交的 token 数从 13
提到 21 就是净赚：

| 模式 | ng12 | ng20 | 提升 |
|---|---:|---:|---:|
| DSpine | 155.4（37 轮） | **239.7**（23 轮） | +54% |
| DFlash2 | 162.6（38 轮） | **218.9**（24 轮） | +35% |
| MTP | 146.7（38 轮） | **202.2**（24 轮） | +38% |

继续加宽在 32 附近摊薄（每轮 126 ms），净收益只剩 +6%。

ng20+ 的测量还明显更稳：冷/暖两次几乎重合（239.9/239.7、218.9/218.9、202.9/202.2），
而 ng12 的冷/暖能差 5–20%（152.1/152.1、189.4/158.5、172.3/165.4 反复横跳）。轮次越少，
页布局和时钟抖动对总时间的摊薄越明显。

## 5. 采纳建议

- **128k 及以上的长输出：`--ngram-draft-tokens 20`**（C1）；命中率极高时可试 24–32。
- **工具调用/agent：保持 12**。宽窗口在 ngram miss 时每轮要白读更多 KV 列，短输出场景
  收益不足以抵消；agent 路径另有 DSpine 结构化输出 grammar 违规的独立问题。
- **不要再用单次 decode 数字做模式对比**：这块卡在 128k 顶功耗墙，单次 ±15%；
  比模式至少交错跑两轮取中位。
- 复现脚本：`C:\temp\dspine_bench\` 下 `bench_variant.py`（单变体）、`ab_cycle.py`
  （交错多轮 + 全程 GPU 采样）、`gpu_sample.py`、`probe_startup.py`（只读启动显存行）。

## 6. 本次未做的事

- 未改代码。`draft.cpp` 里 `safe_window = std::max(512, ring_capacity)` 与上方注释
  「Cap the default at capacity/8」仍然矛盾，但 batch=1 + ngram 的轮次不走这段，属于
  遗留文案问题，不影响本次结论。
- 没有解锁功耗上限（需要管理员且会改用户 GPU 设置），所以 ±15% 的环境噪声无法进一步压缩。
- 没有测 C2–8：并发门禁会把宽度降到 15，本轮全部是 C1。
