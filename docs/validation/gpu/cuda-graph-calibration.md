# 为什么现在不用加 `--no-cuda-graph`

## 一句话

以前关图，是因为引擎按拓扑盲扣约 **1 GiB** 给 CUDA Graph；现在启动时实测只有 **90–143 MiB**，按 **256 MiB** 规划，图照开，KV 还能给到 77,824。所以日常不再需要 `--no-cuda-graph`。

---

## 1. CUDA Graph 到底占什么显存

decode 一步会提交几十个 kernel（RMSNorm、QKV、attention、conv、MLP…）。CUDA Graph 把这些 launch 录成一张图，之后一次提交、整图回放，减少 CPU launch 开销，decode 大约快 6%。

代价是 Graph 本身要占 driver/runtime 显存：

- graph executable；
- 每个 topology 的 driver/module 状态；
- kernel code / 延迟加载的模块；
- Graph 捕获期间的临时缓冲。

这些不在 KV/workspace 的显式 arena 里，所以引擎必须在规划 KV 时单独扣一块。

---

## 2. 旧方案：按拓扑盲预留

旧代码在 `planning/startup.cpp` 里按 `topology_class` 算一个 allowance：

```cpp
// DFlash2：每个 topology class 96 MiB
return (final_visible <= 4096 ? 64ULL : 96ULL) * kMiB;
```

DFlash2 有 6 个 class（96 / 511 / 2047 / 8191 / 32767 / 窗口），再叠加 ngram window，实际会扣到 **约 1 GiB**。

然后在 KV 规划前直接减掉：

```text
可用显存 = 总可用 - 权重 - workspace - graph_allowance
KV 容量  = 可用显存 - ...
```

问题是这是**上界**：

- 不知道实际驱动、CUDA 版本、拓扑会占多少；
- 多个 topology 共享 driver/module 状态，逐类相加会重复；
- 扣多了 KV 就少，36K 以上很容易启动失败。

用户为了把容量拿回来，只能：

```bat
--no-cuda-graph
```

把 1 GiB 还给 KV，代价是 decode 慢约 6%。

---

## 3. 新方案：启动时实测

现在的流程：

1. **权重加载后**，用最小物理 KV 池构造一个探针 Program。
   - 逻辑容量仍是 KVMem 窗口，保证所有 graph profile 都能覆盖；
   - 物理页只分配该窗口需要的页数，不占满 max-context；
   - 探针不参与服务，构造完立即销毁。

2. 探针捕获**完整 graph 集合**：
   - 普通 decode；
   - DFlash2 forward；
   - DFlash2 finish；
   - 所有 topology class。

3. 用 `cudaMemGetInfo`（WDDM 下再用 residency dedicated bytes）测真实占用：

   ```text
   graph_measured_bytes = free_before - free_after
   ```

4. 销毁探针，同步设备。释放后仍保留的 driver 内存已经进入新的 free 基准，**只扣一次**。

5. 正式容量规划：

   ```text
   graph_allowance = max(measured, 256 MiB)
   KV 预算 = available - graph_allowance
   ```

6. 正式 Program 构造后按实际占用检查；探针失败才回退旧盲预留。

实测：

```text
probe measured 90.0–143.5 MiB
planning allowance 256 MiB
```

对比旧方案约 1 GiB，直接省出 **700–800 MiB**。

---

## 4. 为什么是 256 MiB 下限

两个原因：

1. **WDDM 会夹 free delta**：显存吃紧时 `cudaMemGetInfo` 的 free 可能被夹到 0，探针成功但 measured=0。这时不能退回 1 GiB 盲预留，用 256 MiB 保守下限。
2. **给 driver 留余量**：实测是 90–143 MiB，但不同驱动/桌面占用会有波动；256 MiB 留一倍余量。

探针异常（构造失败、OOM）才回退盲预留。

---

## 5. 为什么不再需要关图

以前：

```text
盲扣 ~1 GiB → KV 只够 36K → 想加 KV 只能 --no-cuda-graph
```

现在：

```text
实测 ~100 MiB → 按 256 MiB 规划 → 图照开，KV 到 77,824
```

q3blocks + DFlash2 draft 5 + 开图：

| 配置 | 总容量 | free | 8.7K decode |
|---|---:|---:|---:|
| 开图 + 自动校准 | 77,824 | 457 MiB | 66.3–66.7 tok/s |
| 关图 | 约 90,112 | — | 慢约 6% |

也就是说，关图现在只能多换约 4K token，却要付出 6% 的 decode，不划算。

---

## 6. 什么时候仍然值得关图

| 场景 | 建议 |
|---|---|
| 想压到 90K 以上极限 | 可以关，换约 4K token |
| 探针本身 OOM | 关图或降 B/R |
| 调试 Graph 相关问题 | 关图排除因素 |
| 日常 52K–86K | **不用关** |

---

## 7. 保留的手动开关

- `--cuda-graph-allowance-mib N`：手动指定 allowance，跳过探针；调试或特殊驱动时用。
- `--no-cuda-graph`：完全关闭 Graph，跳过探针。
- 不带这两个参数：自动校准。

---

## 8. 结论

- 旧方案是“按最坏拓扑先扣 1 GiB”，保守但浪费；
- 新方案是“启动时真跑一遍完整 Graph 集合，量出 100 多 MiB，按 256 MiB 规划”；
- 图照开，KV 也够，所以日常不需要 `--no-cuda-graph`；
- 只有在追求 90K+ 极限、探针 OOM 或调试时才关。
