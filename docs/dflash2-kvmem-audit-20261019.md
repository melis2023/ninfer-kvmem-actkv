# DFlash2 × KVMem 代码审查与修复记录（2026-10-19）

> 范围：**纯代码审查（不含实测）** + 落地 F1 / F2 / F3 / F5 修复。
> 审查对象：`dflash2-kvmem-fixes-20261018.md` 之后的代码状态（含 P0-1 / P0-2 / P0-3、
> `dflash_envelopes` 第三参数、floor/slot 生命周期补丁）。
> **回滚依据见第 4 节（改动前原文）与第 5 节（回滚步骤）。**

## 1. 审查结论总览

| 编号 | 位置 | 性质 | 会导致什么问题 | 严重度 | 本轮 |
|---|---|---|---|---|---|
| F1 | `types.h:514`（`recent_tokens = 64`）+ `kvmem_window.cpp:580-583` | floor 保留量（64）与环容量（2048）差 32 倍 | 每次重选后草稿只剩 3.1% 上下文，接受率被钉在 ~30-39%（短上下文 ~44%） | 高 | **已修** |
| F2 | `hybrid_program.cpp:886-891`、`materialization.cpp:623-627`、`prefill.cpp:697` 等；`decode.cpp:957` 无守卫 | frontier 18 个写点 vs floor 5 个写点，floor 无配对的"随 frontier 下降"机制 | floor > frontier 时内核全掩码 → 接受率 ≈0、~24 tok/s，**无任何日志** | 高 | **已修** |
| F3 | `checkpoint_recovery.cpp:770`、`kvmem_window.cpp:11` | 页大小三处独立硬编码 64 | 改 `kPagedKVPageSize` 后 floor 与真实页边界错位 → 静默读错/多掩 | 低 | **已修** |
| F4 | `graphs.cpp:559-619` | `draftless_graphs` = 6 × `max_concurrency` | 图 executable 数量翻倍，挤压启动预留 | 中（成本） | 不改 |
| F5 | `decode.cpp:104` / `:1230` / `:1045-1046` | 注释写 "k = 0"，实现是 k=1 / width=2 | 维护陷阱（按注释删特例分支会崩） | 低 | **已修（注释）** |
| F6 | — | 止损轮是否在环里留空洞 | **不成立，撤回**（见 §3.2） | — | 撤回 |

## 2. 本轮落地改动

### 2.1 F1：floor 保留量改为环容量（`kvmem_window.cpp`）

**机制**：`dflash_context_floor` 是"环内仍可信的最小 logical 位置"，内核按
`context_count = min(max(v − floor, 0), window_mask)` 裁剪可见范围（`window_mask = S − 1`）。
`S = draft.sliding_window = 2048` 是 DFlash2 conditioning 环的容量。
原代码 `floor = end − recent_blocks()*64 = end − 64` → **环里 2048 行只剩 64 行可见**。

**修复**：重选路径的 floor 保留量取 `max(kvmem recent 页, 环容量)`。
`floor = end − 2048` 时 `v − floor = 2048 = S` → 内核 clamp 到 `S − 1` → **整环可见**，
即等价于"重选不再掩码"，但仍保留"环容量不足时按实际容量保守"的语义。
`KvmemOptions::recent_tokens`（页保留策略）**不动**，只改 floor 计算。

> 依据：环（`CyclicKVCache local`，`DFlashPersistentState.local`）是**独立设备分配**，
> 每轮由 `append_context_impl`（`draft.cpp:621`，无条件、在 draftless 判定之前）追加，
> KVMem 的 paged 块换出**不会**使环行失效。因此重选路径上的 floor 抬升是过度保守的。
> 这也与 `20261018.md` §5.3 自述一致："`recent_tokens >= 2048`（= 环容量）才能零上下文损失"。

**恢复路径（`kvmem_window.cpp:1181`、`checkpoint_recovery.cpp:771`）本轮保持原值**
（ring snapshot 的覆盖范围尚未验证，保守不动；见 §3.4）。

### 2.2 F2：floor ≤ frontier 不变式（`context.cpp` + `decode.cpp`）

**机制**：`dflash_context_frontier` 有 18 个写入点，`dflash_context_floor` 只有 5 个
（3 处抬高 + 2 处复位为 0）。**没有任何一条路径保证 frontier 重设时 floor 同步下调**。
明确暴露的路径：

- `hybrid_program.cpp:884-892`：`reuse == 0` 才走 `ordered_reset`（floor = 0）；
  `reuse != 0`（连续对话下一轮、前缀复用）**只设 frontier** → floor 保留上一请求终值。
  若 `reuse < floor` → 内核全掩码 → 接受率 ≈0。
- `materialization.cpp:623-627`：`source.dflash_context_frontier = details.reuse_base`，
  同样不碰 floor。
- `prefill.cpp:697`（`PrivateLongAnchor`）：`= base`，同样不碰 floor。

原 `decode.cpp:957` 只断言 `dflash_context_frontier > execution_frontier`，**没有 floor 守卫**，
所以故障是静默的（症状与 20261018 的原始 bug 无法区分）。

**修复**：

- 新增 `ProgramImpl::clamp_masked_draft_floor(SequenceState&) noexcept`：
  `floor = min(floor, frontier)`（超过则夹到 frontier），并在
  `refresh_state_views()` 末尾调用 —— 该函数被所有"请求开始 / 恢复 / 材料化 / prefill
  chunk"路径调用，正好覆盖上述暴露点。
- `decode.cpp` 的 dflash decode-ready 循环内（每轮）再夹一次作为兜底自愈，
  保证任何遗漏路径不会静默全掩码。

### 2.3 F3：页大小收敛到 `kPagedKVPageSize`

`kvmem_window.cpp:11` 与 `checkpoint_recovery.cpp:770` 各自硬编码 64，
而权威常量是 `core/paged_kv_cache.h:19` 的 `kPagedKVPageSize`（`kv_pages_for_tokens` 用的就是它）。
三处独立硬编码同值，一旦页大小调整即错位。本轮把两处改为引用 `kPagedKVPageSize`（值不变，行为不变）。

### 2.4 F5：注释与实现对齐

`decode.cpp:104` 与 `:1230` 的 "k = 0" 描述与 P0-1 实现不符。实际实现是
**k = 1 / verify width = 2**（`neural_proposal_drafts = 0` + `family_window = 1` +
`proposal_valid_columns = 2` + `target_valid_columns = extent+1 = 1`）。
另外更正一处此前的错误判断：`proposal_valid_columns`（`decode.cpp:1045-1046`）
**不是死码**，其消费者是 `draft.cpp:281`（`propose_dflash2_batch` 内）。
只是在 draftless 轮该值不被消费（`draft.cpp:637` 的 `!draftless` 跳过了提议），
所以 `2U` 是无害冗余，**不应删除**。

## 3. 未落地 / 撤回项

### 3.1 F4（不改，仅记录）

`draftless_graphs`（`graphs.cpp:559-619`，P0-2）规模 = profile 桶数(6) × `max_concurrency`。
默认 `max_concurrency = 1` → +6 个 executable（与 neural 族 6 个等量，DFlash 图总量翻倍）；
8 路服务时 +48。保留该族是正确的（宽度不同会在同一 topology 槽互相驱逐）。
`IQ3_S-mtp` 启动失败（`Engine runtime reservation requires 2952544512 bytes`）
**不能归因**到该改动（模型 13.16 GiB + 桌面占用本身已超额）。

### 3.2 F6 撤回

此前存疑"止损轮不写环 → 重探针轮读到空洞"，核查后**不成立**：

1. `include/ninfer/ops/sliding_window_attention.h:36`：cyclic context 是 **read-only**
   （"stores BF16 K and V for committed absolute positions `[max(0,L-window),L)`"）。
2. 环写入者是 `append_context_impl`（`draft.cpp:93` 定义 / `:621` 调用），
   经 `kv_cache_append_prefix` 写入环。
3. `draft.cpp:618-623` 的 append **无条件执行且位于 draftless 判定（`:628-629`）之前**
   → 止损轮同样写环。
4. 写入列数由 `target_valid_columns` 门控：止损轮 `extent = 0` → `target_valid_columns = 1`
   → 恰好补上 anchor 位置本身，与上一轮已提交位置连续。

结论：环由每轮无条件维护，位置连续无洞。

### 3.3 恢复路径 floor 待验证（本轮保守未动）

`kvmem_window.cpp:1177-1182` 与 `checkpoint_recovery.cpp:767-772` 仍用
`recent_blocks()*page_tokens`（= 64）。这两处对应 **StateImage ring snapshot 的恢复**，
其必要性取决于快照实际覆盖多少环行：

- 若快照覆盖整环（2048 行）→ 可安全提到环容量（额外收益）；
- 若只覆盖尾部若干行 → 保持 64 是正确的（提到 2048 会读到残留/其它序列的环行）。

需要确认 `state_images` 布局里 `dflash_local_layer_bytes` 的覆盖范围后，
再决定是否与重选路径统一（见 `20261018.md` §5.3 的同类结论）。

### 3.4 未覆盖的检查项

- 本轮**未做任何实测**（按用户要求）。F1 的收益预期（64k 越窗接受率 30-35% → ~44%）
  与 F2 的可达性都需重启服务后实测确认。
- `P1`（prefill 越窗掉速 ~870 → ~390 tok/s）、`P3`（`--kvmem-gen-reserve` / budget 配比）、
  `P4`（ngram 长文档拷贝）、`P5b`（自适应宽度）、`P6`（温度）仍未落地。

## 4. 改动前原文备份（回滚依据）

### 4.1 `backends/ninfer/src/models/qwen3_5/program/program_impl.h`

```cpp
    // Masked-draft conditioning floor: the smallest logical position whose feature is still
    // guaranteed live in the DFlash2 cyclic ring. A KVMem reselection/compaction swaps selected
    // blocks except the trailing recent pages, so this rises to the start of that trailing
    // continuous segment; rounds mask older ring slots so stitched history cannot poison the
    // draft. DFlash's Full pool is unaffected.
    std::uint32_t dflash_context_floor = 0;
```
（`SequenceState`，原 :614-619）

```cpp
    void refresh_state_views(SequenceState& sequence);
    void reserve_state_entitlement(SequenceState& sequence, std::uint32_t slots);
```
（原 :1819-1820）

### 4.2 `backends/ninfer/src/models/qwen3_5/program/storage/context.cpp`

```cpp
void ProgramImpl::refresh_state_views(SequenceState& sequence) {
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
    if (state_store->valid(sequence.state.read) && state_store->valid(sequence.state.write) &&
        state_store->residency(sequence.state.read) != StateReplicaResidency::HostOnly &&
        state_store->residency(sequence.state.write) != StateReplicaResidency::HostOnly) {
        const StateImageHandle committed =
            sequence.state.fork_pending ? sequence.state.read : sequence.state.write;
        sequence.tail_hidden =
            state_images->continuation_hidden_slot(state_store->physical_slot(committed));
    }
    if (sequence.rewrite_state && state_store->valid(*sequence.rewrite_state) &&
        state_store->residency(*sequence.rewrite_state) != StateReplicaResidency::HostOnly) {
        sequence.rewrite_checkpoint_hidden = state_images->continuation_hidden_slot(
            state_store->physical_slot(*sequence.rewrite_state));
    }
}
```
（原 :1167-1183，`clamp_masked_draft_floor` / `masked_draft_ring_capacity` 定义插入其前）

### 4.3 `backends/ninfer/src/models/qwen3_5/program/storage/kvmem_window.cpp`

```cpp
constexpr std::uint32_t page_tokens = 64;
```
（原 :11）

```cpp
    if (force_selection || compact_position(sequence, end) > kvmem_window_tokens) {
        // A reselection replaces the selected blocks; only the trailing recent pages are
        // guaranteed to stay. Raise the masked-draft floor to that trailing continuous segment
        // so the DFlash2 ring stops reading features of blocks that were swapped out.
        const std::uint64_t recent_tokens =
            static_cast<std::uint64_t>(kvmem_options.recent_blocks()) * page_tokens;
        sequence.dflash_context_floor =
            end > recent_tokens ? static_cast<std::uint32_t>(end - recent_tokens) : 0U;
```
（原 :576-583）

### 4.4 `backends/ninfer/src/models/qwen3_5/program/planning/checkpoint_recovery.cpp`

```cpp
        const std::uint64_t recent_tokens =
            static_cast<std::uint64_t>(kvmem_options.recent_blocks()) * 64ULL;
```
（原 :769-770）

### 4.5 `backends/ninfer/src/models/qwen3_5/program/decode.cpp`

```cpp
// A bypassed request runs draftless rounds (k = 0: target-only verification, no drafter). The
// content that made the drafter hopeless can give way to something predictable later in the same
// request, so every this many draftless rounds the drafter is given one more chance.
```
（原 :104-106）

```cpp
        SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
```
（原 :948-950，夹紧调用插入 `const RequestControl& request` 之后）

```cpp
    // A request whose neural drafter accepted nothing for a streak is logged (KVMEM_DFLASH_BYPASS)
    // and then runs draftless rounds inside the dflash batch (k = 0, target-only verification, no
    // drafter): the ordinary path has no execution resources under a masked-draft backend.
```
（原 :1229-1231）

## 5. 回滚步骤

1. 按第 4 节逐条把 5 个文件恢复为原文（或 `git checkout -- <file>`，
   若仓库已纳入版本控制）。
2. 本文件本身可保留（不改动运行时行为），仅第 2 节失效。
3. 重建：

```powershell
call "D:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
set CUDACXX=<CUDA>\v13.4\bin\nvcc.exe
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
cmake --build build-backends\ninfer --target ninfer-serve
```

4. 部署（停服务后校验 `SRC == DST`）：

```powershell
Copy-Item build-backends\ninfer\apps\ninfer-serve.exe workers\ninfer\ninfer-kvmem-server.exe
```

## 6. 备注

- 本轮全部改动均为**行为等价或保守化 + 一处参数语义修正**（F1），不改变 KVMem
  的页保留策略、不改变图族结构、不改变 ingress 布局。
- F1 修复后，`--kvmem-recent-tokens` 可以继续用默认值；若想显式对齐，加
  `--kvmem-recent-tokens 2048` 亦可（但会同时把 KVMem 强制保留的最近页从 1 页提到 32 页，
  非必要不建议）。

## 7. 实测验证（2026-10-19 02:4x）

**环境**：修复后 exe SHA256 `F7CA4DB0…52EF`（部署到 `workers\ninfer\ninfer-kvmem-server.exe`，
hash 校验一致）；模型 `I:\model\gguf\Qwen3.8-27B-GSQ-RCO-IQ3_XXS-dflash2.ninfer`；
窗口 B+R = 53248；payload `sweep.json`（greedy，重复文本，prompt 59336 **越窗**）。

| 场景 | P0 基线（修复前） | F1 修复后 | 变化 |
|---|---:|---:|---|
| 64k 越窗，`--draft-tokens 4` | 108/359 = **30.1%**，decode 45.9–49.2，prefill 375–393 | 114/337 = **33.8%**，decode 47.3，prefill 376.4 | +3.7 pp |
| 64k 越窗，`--draft-tokens 3` | 102/288 = **35.4%**，decode 47.7，prefill 388–395 | 109/268 = **40.7%**，decode 43.4，prefill 380.2 | **+5.3 pp** |
| t2 紧随 64k 之后（dt=4） | 230/671 = 34.3%（`t2-after-64k`） | 244/614 = **39.7%**，decode 56.1 | +5.4 pp |
| t2 紧随 64k 之后（dt=3） | 1.4%（20261018 版）/ 44.7%（warm） | 237/483 = **49.1%**，decode 54.9 | — |

**判读**

- 越窗场景的接受率已**对齐同参数的"无 floor 掩码"基线**：dt=4 为 33.8% vs 短上下文
  `warm-t2` 34.3%；dt=3 为 40.7%（短上下文 44.7%，剩余差距来自 KVMem 压缩对目标模型
  上下文的影响，与 floor 无关）。这证明 F1 的 floor 抬升确已消除。
- `KVMEM_DFLASH_BYPASS` / `KVMEM_DFLASH_FORK` 全程 0 行，无 ERROR/FATAL。
- prefill 仍为 ~376–380 tok/s（越窗后腰斩问题不变，属 P1，未在本轮范围内）。
- **dt=3 优于 dt=4**（40.7% vs 33.8%，decode 差异在噪声内），与 `20261018` 时期
  "dt=3 吞吐最优"的结论一致；bat 现为 `--draft-tokens 4`（用户指定），可按需回退到 3。

**仍未验证**（需后续实测）

1. F2 的可达路径（同槽前缀复用 / materialization 截断）：本轮未构造出触发场景，
   修复为防御性夹紧 + 自愈，未观察到告警。
2. 恢复路径 floor（`kvmem_window.cpp:1181` / `checkpoint_recovery.cpp:771`）仍为 64，
   若确认 ring snapshot 覆盖整环，可与重选路径一并提到环容量（额外收益）。

