# KVMem 差异 restore：只搬变化的页（2026-10-20）

## 1. 问题

每次重选都会执行一次全量 restore：

```cpp
destructive_truncate(seq.kv->text, 0);          // 清空全部映射
ensure_mapped_to_tokens(seq.kv->text, compact_end);
copy_from_host_records(restore_records, ... destinations ...);  // 576 页全量 H2D
```

`KVMEM_RESTORE pages=576 read_us=65000 sync_us=94000`，每次约 160 ms。

但重选后的选中集合通常和上一轮高度重叠：576 个位置里只有 100–200 个逻辑页真正变了，其余 400 多页的数据本来就在显存里，被 `truncate(0)` 清掉后又从 host 搬了一遍。

参考实现（`paicat1/Ninfer-KVMem-Async`）在 restore 时直接跳过 `device_resident` 的页，只搬差异。它之所以能这样做，是因为选中页的 physical slot 在重选之间保持稳定。

## 2. 实现

`backends/ninfer/src/models/qwen3_5/program/storage/kvmem_window.cpp`

### 2.1 选中集稳定排序

```cpp
for (const auto& block : plan.next.blocks) selected.push_back(id);
// Stable page order is what makes a slot-by-slot diff meaningful.
std::sort(selected.begin(), selected.end());
```

`plan.next.blocks` 的顺序来自检索结果，不保证按页 id。排序后，同一个逻辑页在重选后仍落在同一个 physical slot。

### 2.2 和上一轮前缀逐位置比较

```cpp
const auto& previous = seq.window.pages;
const bool enough = previous.size() >= selected.size();
std::vector<std::uint32_t> changed;
if (enough) {
    for (std::uint32_t i = 0; i < selected.size(); ++i)
        if (previous[i] != selected[i]) changed.push_back(i);
}
const bool incremental = !checked && enough &&
    changed.size() * 2 <= selected.size() &&
    selected.size() >= selection_pages / 2;
```

- `window.pages` 可能比选中集长（尾部 reserve 页），所以只比较前缀；
- `changed` 超过一半时回退全量，避免差异比全量还贵；
- `sel=1` 这类特殊路径（probe 的单页窗口）不进入增量分支；
- `--kvmem-verify-transfers` 的字节校验路径强制全量，保证校验覆盖所有页。

### 2.3 只读、只搬差异页

```cpp
if (incremental) {
    for (const std::uint32_t i : changed)
        record(selected[i])->read(0, staging + i * page_bytes, page_bytes);
} else {
    for (std::size_t i = 0; i < selected.size(); ++i) ...
}
```

H2D 同样只对 `changed` 位置：

```cpp
std::vector<DeviceKVPageHandle> destinations;
std::vector<const std::byte*> records;
for (const std::uint32_t i : slots) {
    destinations.push_back(p.text_kv_addresses->physical_page(seq.kv->text, i));
    records.push_back(restore_records[i]);
}
pool.copy_from_host_records(records, {}, destinations, backend.layout, p.compute_streams);
```

### 2.4 增量路径不再清空映射

```cpp
destructive = !incremental;
if (!incremental) {
    destructive_truncate(seq.kv->text, 0);
} else if (previous.size() > selected.size()) {
    // 只截掉尾部 reserve 页，保留前缀映射
    const auto frontier = static_cast<std::uint32_t>(plan.next.visible_tokens);
    destructive_truncate(seq.kv->text, frontier);
}
```

`frontier` 用 `plan.next.visible_tokens`，不是 `selected.size() * page_tokens`：最后一页可能是 partial，`commit_frontier()` 会拒绝更小的值。

MTP 的 backend address 做同样处理。

## 3. 实测（64K，8 次重选）

| 重选 | changed | restore 页数 | read_us | sync_us | 总耗时 |
|---|---:|---:|---:|---:|---:|
| 1 | 560 | 576（回退全量） | 69,044 | 94,021 | 163 ms |
| 2 | 560 | 576（回退全量） | 66,202 | 94,045 | 160 ms |
| 4 | — | 1（特殊路径） | 238 | 18 | 0.3 ms |
| 5 | 124 | **124** | **13,564** | **19,895** | **33 ms** |
| 6 | 164 | **164** | **20,234** | **26,570** | **47 ms** |
| 7 | 124 | **124** | **14,616** | **20,020** | **35 ms** |
| 8 | 181 | **181** | **21,381** | **29,369** | **51 ms** |

稳定阶段：restore 从 160 ms 降到 33–51 ms，**约 3.5–4.8 倍**。

前几次 `changed=560` 是因为初始 prefill 的页布局与第一次重选的选中集差异很大，自动走全量。

64K 端到端：prefill 422.4 tok/s、decode 49.3 tok/s、draft 接受率 51.3%。

## 4. 256K NIAH

召回与基线完全一致：

| 指标 | 基线 | 差异 restore v2 |
|---|---:|---:|
| same-turn | 33/33 | **33/33** |
| final strict | 33/33 | **33/33** |
| final union | 33/33 | **33/33** |
| cross-tool | 14/14 | **14/14** |
| 工具调用 | 80 | **80** |
| 端到端（到 final sweep） | 637.1 s | **594.8 s** |
| write prefill | ~390 tok/s | 368–976 tok/s |
| probe decode | 49.5 tok/s | 57.9–81.2 tok/s |

**但 256K 的收益不是差异 restore 带来的。**

178 次 restore 里只有 4 次走增量路径：

| 项 | 值 |
|---|---:|
| restore 总次数 | 178 |
| 全量（576 页） | 170 |
| 增量 | 4 |
| `changed` 中位数 | **495 / 576** |
| `changed` 平均 | 290 |
| reselect 传输总和 | 19.41 s |

`changed` 中位数 495 意味着重选后 576 个位置里有 495 个逻辑页变了，超过一半，自动回退全量。原因是 80% 水位重选时，窗口从 42.6K 回落到 36.8K，检索 top-K 在高竞争（250K 里选 36K）下几乎整体换血；只有 sink 的 16 页位置稳定。

64K 稳定期 `changed` 只有 124–181，是因为上下文（59K）只略微超过窗口（53K），选择集变化小。

所以：

- **64K**：差异 restore 有效，restore 160 ms → 33–51 ms；
- **256K**：差异 restore 自动回退，restore 仍约 160 ms，端到端 594.8 s 与基线 609.5–637.1 s 的差异来自 prefill 波动（368–976 tok/s），不能归因于本次改动。

要让 256K 也受益，需要按逻辑页 id 做集合匹配 + 显存内 D2D 搬移，而不是按位置比较。那是另一个量级的改动。

## 5. 三个坑

1. **`window.pages` 有 664 页**（576 选中 + 88 reserve 尾），长度直接比较失败 → 改成前缀比较；
2. **`std::equal` 全等判断太严**，`changed` 只在全等时才计算，等于没测到差异 → 改成只看长度、逐位置统计；
3. **`destructive_truncate` 会重置 `committed_frontier`**，`publish` 时 `commit_frontier(view.visible_tokens)` 如果更小就会抛 `KV committed frontier is invalid` → trim 的 frontier 改用 `plan.next.visible_tokens`。

## 6. 回退

```bash
cd backends/ninfer
git checkout -- src/models/qwen3_5/program/storage/kvmem_window.cpp
```

或从 `I:\llama\_backup\kvmem-llama.cpp-multi-backend-framework-20261020-143834` 恢复。

## 7. 后续

- 增量路径的 `remap_us` 已经降到 100 µs 以下（不 truncate 就不重映射）；
- 剩余成本主要是 `sync_us`（H2D 等待），与 `changed` 页数成正比；
- 可以继续做异步 spill（`PendingDemote` 那一套），但 restore 必须同步，因为下一步立即要用选中的 KV。
