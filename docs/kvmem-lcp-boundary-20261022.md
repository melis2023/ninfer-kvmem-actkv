# LCP 边界函数移植（2026-10-22）

## 0. 结论

从上游 `1ffa48a` 取入纯新增的 LCP（Longest Common Prefix）边界逻辑，**未接入现有调用路径**，只作为公共能力存在。

| 项 | 状态 |
|---|---|
| `prefix_checkpoint.hpp` | 新增，141 行 |
| `prefix_checkpoint.cpp` | 新增，54 行 |
| `prefix_checkpoint_test.cpp` | 新增，通过 |
| 现有代码 | **未改动** |
| 编译 | sm120a 0 错误 |
| 回归 | `mean_key_retrieval_test`、`kvmem_store_test` 通过 |

## 1. 为什么补这个

现有 `memory_same_query`（`kvmem_window.cpp:984`）的复用判定是：

```cpp
if (!query_tokens || window.query_tokens != window.query_end - window.query_begin ||
    frontier < window.query_end) return false;
return query.begin == window.query_begin && query.count == window.query_tokens;
```

两条约束：query 完全一致、frontier 已越过 query 尾部。

上游的 LCP 逻辑多了两条防御：

1. **不能恢复到已评估边界之后**：`min(lcp, live, eval_end - needs_logits)`
2. **不能把不可分割的输入组切开**：`prefix_anchor` 会向回退到组边界

这两条针对的是「复用后输出错乱」——恢复位置落在了还没算完 logits 的地方，或切断了图片/tool call 这类连续结构。

## 2. 三个函数

```cpp
// 能恢复到哪
uint64_t prefix_restore_limit(uint64_t lcp, uint64_t live,
                              uint64_t eval_end, bool needs_logits);

// 复用锚点，避开不可分割区间
uint64_t prefix_anchor(uint64_t base, uint64_t lcp, uint64_t query,
                       const std::vector<PrefixRange> & indivisible = {});

// checkpoint 保留策略（按用途优先级 + 字节预算）
std::vector<size_t> retain_checkpoints(const std::vector<CheckpointInfo> &,
                                       const CheckpointLimits &);
```

`prefix_anchor` 的核心逻辑：

```cpp
uint64_t anchor = std::clamp(lcp, base, query);
do {
    changed = false;
    for (const auto & range : indivisible) {
        if (range.begin < anchor && anchor < range.end) {
            if (range.begin < base) throw ...;
            anchor = range.begin;
            changed = true;
        }
    }
} while (changed);
```

即：锚点落在某个不可分割区间内部时，回退到该区间起点，重复直到稳定。

## 3. 接入方式（未做）

如果要让这两个函数生效，需要改 `memory_same_query`：

```cpp
// 现在
if (frontier < window.query_end) return false;

// 接入后
const auto limit = kvmem::prefix_restore_limit(
    lcp, window.query_tokens, frontier, /*needs_logits=*/true);
if (limit < window.query_end) return false;
```

以及把 `window.pages` 对应的不可分割区间传给 `prefix_anchor`。

**这一步没有做**，因为：

1. 需要先确认 `lcp` 在你的代码里对应哪个量（`memory_query.count`？`text_kv_valid`？）
2. `indivisible` 区间需要从 token 边界推导（工具调用、图片、针列表）
3. 改错了会导致复用更保守，召回不受影响但速度下降

## 4. 测试覆盖

`prefix_checkpoint_test` 验证：

- 重建检查点的用途分类
- 保留策略（按优先级 + 字节预算，别名只计一次）
- 租约与边界（`prefix_restore_limit` 的裁剪、`prefix_anchor` 的回退）
- 非法输入（`query < base`、区间反向、区间切到 base 之前）

## 5. 后续

| 步骤 | 内容 | 风险 |
|---|---|---|
| 1 | 确认 `lcp` 在 `kvmem_window.cpp` 的对应量 | 低 |
| 2 | 先在 `memory_same_query` 里加 `eval_end` 约束 | 低 |
| 3 | 再考虑 `indivisible` 区间 | 中（需要 token 边界推导） |
| 4 | 跑 256K NIAH + LongMemEval 确认无回归 | — |

**当前状态是「能力已就位，未启用」**，不影响现有行为。
