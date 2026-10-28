# DSH 适配：统一 keep 开关、工具参数顺序、空参数过滤（2026-10-28）

本文档记录为 DeepSeek Harness（DSH）场景做的 KVMem / NInfer 代码改动，便于后续恢复、对比和回退。

> 状态：**工作区改动，尚未编译**。分支 `exp/auto-pin`，基于 `main`（`69864b0`）。
> 旧版 L1 文档：`docs/kvmem-l1-retention-20261026.md`。

---

## 1. 目标

DSH 的 agent 会话有几个典型问题：

1. 工具 schema 很大（38 个工具约 11K token），固定 8% 的 tools pin 只能保留约 48 页，会截断；
2. 首个 user brief 固定 4%，也可能不够或浪费；
3. 系统提示词没有单独保留；
4. 上一条回答的 thinking 以外的尾部需要保留，但固定 20% 可能挤压检索；
5. 模型偶尔生成空的 tool-call 参数 `{}`，DSH 逐个执行，产生大量 `ToolArgsError: INVALID_ARGS`。

本次改动把 1–4 合并到一个开关，并在服务端过滤空参数调用。

---

## 2. 统一开关：`--kvmem-keep`

### 2.1 参数

```text
--kvmem-keep                 开启结构保留
--kvmem-keep-ratio N         answer 尾上限，占 B 的百分比，1..50，默认 20
--kvmem-pin-answer           别名，等价于 --kvmem-keep
--kvmem-pin-ratio N          别名，等价于 --kvmem-keep-ratio
```

旧 bat 不需要改：

```bat
--kvmem-pin-answer --kvmem-pin-ratio 20
```

新写法：

```bat
--kvmem-keep --kvmem-keep-ratio 20
```

### 2.2 开启后接管

| 集合 | 来源 | 上限 |
|---|---|---|
| system | prompt 开头到 `tools_region.begin` 之前的页 | B 的 10%（无 tools 时 40%） |
| tools | `tools_region` 实际页数 | B 的 40% |
| brief | `first_user_region` 实际页数 | B 的 8% |
| answer | 上一条回答 `thinking_end..frontier` 的页 | `keep-ratio`，且不超过剩余预算 |

总 mandatory 上限 **65% of B**，检索至少保留 35%。

### 2.3 关闭后

不传 `--kvmem-keep` / `--kvmem-pin-answer` 时：

- 不 pin system/tools/brief；
- 不 pin answer；
- 回到只有 sink/recent/media 的旧行为。

这是本次的关键变化：`retain_prompt_spans` 现在必须 `keep` 才生效，不再因为 KVMem 开启就自动 pin。

---

## 3. 代码改动

### 3.1 `backends/ninfer/include/ninfer/types.h`

```cpp
struct KvmemOptions {
    // One switch for structural retention: system preamble, tool schema,
    // first user brief and the non-thinking answer tail.
    bool keep = false;
    std::uint32_t keep_answer_percent = 20;
    ...
};
```

原来的 `pin_answer` / `pin_answer_percent` 已重命名为 `keep` / `keep_answer_percent`。

### 3.2 `backends/ninfer/src/serve/serve_options.cpp`

```cpp
if (arg == "--kvmem-keep" || arg == "--kvmem-pin-answer") {
    options.kvmem.keep = true;
    continue;
}
if (arg == "--kvmem-keep-ratio" || arg == "--kvmem-pin-ratio") {
    const auto percent = parse_nonnegative_int(
        require_value(arg.c_str()), arg.c_str());
    if (percent < 1 || percent > 50) {
        throw std::invalid_argument(arg + " must be in 1..50");
    }
    options.kvmem.keep_answer_percent = static_cast<std::uint32_t>(percent);
    continue;
}
```

帮助文本已更新为 `--kvmem-keep` / `--kvmem-keep-ratio`，旧参数标注为 alias。

### 3.3 `backends/ninfer/src/models/qwen3_5/program/storage/kvmem_window.cpp`

#### `retain_prompt_spans`

入口：

```cpp
if (!kvmem_window_tokens || !kvmem_options.keep) { return; }
```

自动计算：

```cpp
const std::uint32_t system_end = prompt.tools_region
    ? prompt.tools_region->begin
    : (prompt.first_user_region ? prompt.first_user_region->begin : 0U);
const std::uint32_t system_pages_needed =
    system_end ? (system_end + page_tokens - 1U) / page_tokens : 0U;
const std::uint32_t tool_pages_needed = prompt.tools_region
    ? (prompt.tools_region->count + page_tokens - 1U) / page_tokens : 0U;
const std::uint32_t brief_pages_needed = prompt.first_user_region
    ? (prompt.first_user_region->count + page_tokens - 1U) / page_tokens : 0U;

const std::uint32_t system_cap = std::min(system_pages_needed,
    keep * (prompt.tools_region ? 10U : 40U) / 100U);
const std::uint32_t tool_cap   = std::min(tool_pages_needed, keep * 40U / 100U);
const std::uint32_t brief_cap  = std::min(brief_pages_needed, keep * 8U / 100U);
```

- `system_pages`、`tool_pages`、`brief_pages` 分别保存；
- 日志（`--kvmem-verify-transfers`）：

```text
KVMEM_PROMPT_PIN system=1/1 tools=135/135 brief=14/14 pages=...
```

#### `retain_generated_spans`

入口：

```cpp
if (!kvmem_window_tokens || !kvmem_options.keep || !valid_sequence(sequence)) { return; }
```

answer 上限：

```cpp
const std::uint32_t percent =
    std::min<std::uint32_t>(50U, std::max<std::uint32_t>(1U, kvmem_options.keep_answer_percent));
const std::uint32_t reserve = std::min<std::uint32_t>(keep, 2U);
const std::uint32_t structural = static_cast<std::uint32_t>(
    window.system_pages.size() + window.tool_pages.size() + window.brief_pages.size());
const std::uint32_t available =
    keep > structural + reserve ? keep - structural - reserve : 0U;
const std::uint32_t cap =
    std::min({available, keep * percent / 100U, keep * 35U / 100U});
```

规则不变：

```text
generated_begin .. thinking_end   → demote（软降级）
thinking_end .. frontier          → pin（answer）
```

#### `prepare_window`

mandatory 顺序：

```text
sink/media/recent → system → tools → brief → answer
```

总上限：

```cpp
const std::uint32_t structural_cap = keep * 65U / 100U;
```

超出时从 answer 开始裁，检索至少保留 35%。

### 3.4 `backends/ninfer/src/models/qwen3_5/program/program_impl.h`

`KvmemWindowState` 和 `KvmemPrefixCheckpoint` 新增：

```cpp
std::vector<std::uint32_t> system_pages;
```

checkpoint capture/restore 已经同步。

### 3.5 `backends/ninfer/src/serve/http_server.cpp`

`/props` 输出：

```cpp
{"keep", options.kvmem.keep},
{"keep_answer_percent", options.kvmem.keep_answer_percent},
{"pin_answer", options.kvmem.keep},                 // compatibility
{"pin_answer_percent", options.kvmem.keep_answer_percent},
```

---

## 4. Decode pressure（0.19 边驱逐策略的本地移植）

参数：

```text
--kvmem-decode-pressure          默认开启
--no-kvmem-decode-pressure       关闭
--kvmem-decode-pressure-interval N   默认 16
```

行为：

- 每 N 个 accepted token 采样本轮 query 的最后一层 normalized Q；
- 对 resident 页的 Mean-K 做 cosine，softmax（温度 8），EMA 累积（半衰约 8192 token）；
- 窗口满时 `prepare_window` 走 pressure 分支：
  - `selected = mandatory`（sink/media/recent/system/tools/brief/answer）；
  - 其余 resident 页按 `attention_mass` 排序填充；
  - **不 cold recall**；
- 与 0.19 的差异：
  - 0.19 用 native page attention 的 token log-sum-exp；
  - **当前版本仍是 `accepted-Q × Mean-K` 近似**，不是逐 token/head 的 native attention；
  - native page-mass kernel 尚未实现（需要新增 CUDA op）；当前先验证 pressure 路径和默认开启；
  - `attention_mass` 不随 checkpoint 保存，restore 后重建；
- **默认开启**，可用 `--no-kvmem-decode-pressure` 关闭；
- LongMemEval 输出 1024 < R 时通常不触发，主要用于 DSH 长输出或小 R 场景。

启动示例：

```bat
--kvmem-keep --kvmem-keep-ratio 20
```

## 5. DSH 工具 schema 优化

文件：`backends/ninfer/src/serve/translate.cpp`

### 4.1 短参数前置

`order_tool_schema_properties(schema, tool_name)`：

```text
所有工具: file_path / path / file
pwsh:     description
grep:     pattern
```

`properties` 和 `required` 都按这个顺序重排。JSON Schema 语义不变，只影响 prompt 渲染顺序。

### 4.2 描述强化

`write` / `edit`：

```text
Always provide file_path first; it is required even when the payload is long.
```

`pwsh` / `grep`：

```text
All required arguments must be present; never send an empty object.
```

### 4.3 目的

- 减少“写完长 content 后漏 `file_path`”；
- 减少 `pwsh`/`grep` 只生成部分字段；
- 对完全空的 `{}` 无效，需要下一节的过滤。

---

## 6. 空参数 tool-call 过滤

文件：`backends/ninfer/src/serve/openai_chat_response.cpp`

### 5.1 `materialize_tool_calls`

```cpp
for (const ninfer::GeneratedToolCall& call : generated) {
    // A model that decides to call a tool but emits an empty argument object
    // cannot satisfy any real schema. Drop it so the client never executes an
    // empty call and never turns it into a ToolArgsError.
    if (call.arguments_json.empty() || call.arguments_json == "{}") { continue; }
    calls.push_back(...);
}
```

### 5.2 流式 / 非流式同步

```cpp
const std::vector<ToolCall> calls = materialize_tool_calls(outcome.tool_calls);
const bool has_tool_calls = !calls.empty();   // 非流式
```

```cpp
const std::vector<ToolCall> calls = materialize_tool_calls(outcome.tool_calls);
if (!calls.empty()) { ... } else { ... }      // 流式
```

避免出现“`tool_calls` 空数组但 `finish_reason=tool_calls`”。

### 5.3 效果

- `edit {}`、`write {}`、`pwsh {}`、`grep {}` 不再返回给 DSH；
- DSH 不会执行空调用，不会产生 `ToolArgsError: INVALID_ARGS`；
- 但模型不会收到“参数为空”的明确反馈，可能继续生成空调用；需要 DSH 侧重试限制。

---

## 7. DSH profile 配置

文件：`C:\Users\zhuan\.dsh\profiles\desktop\cordis.patch.yml`

### 6.1 压缩触发

DSH 的 `compaction-basic`：

```text
threshold = floor(min(W × thresholdRatio, W − O − B))
O = 模型 maxTokens
B = headroomTokens
```

当前 `W=262144`、`O=64000`、`B=65536`：

```text
threshold = min(209715, 132608) = 132608
```

所以正文 100K + system/tools/reasoning 约 30K 时触发。

### 6.2 80% 压缩配置

```yaml
- id: llm-pi-ai
  config:
    providers:
      nf:
        models:
          - id: ninfer-27b
            contextWindow: 262144
            maxTokens: 32768

- id: compaction-basic
  config:
    thresholdRatio: 0.8
    headroomTokens: 16384
    retainRatio: 0.16
```

计算：

```text
W − O − B = 262144 − 32768 − 16384 = 212992
threshold = min(209715, 212992) = 209715（80%）
```

注意：`maxTokens` 从 64K 降到 32K 会限制单次输出，长 HTML 更容易分块。

### 6.3 不设置 maxTokens

DSH 默认 `defaultMaxTokens = 32768`，不是无限：

```text
threshold = min(209715, 262144 − 32768 − 65536 = 163840) = 163840（62.5%）
```

只改 `maxTokens` 到不了 80%，因为 `headroomTokens=65536` 单独就超过 `52,429` 的预算。

---

## 8. DSH write 长度限制（未实施）

问题：模型生成 26K/18K/21K 字符的 `write` content，写到一半就闭合 JSON，DSH 仍执行，文件不完整。

推荐改动位置：

```text
C:\Users\zhuan\.dsh\profiles\node_modules\@deepseek-ai\dsh-tool-fs\lib\index.js
write.execute
```

代码：

```js
const maxWriteChars = Number(process.env.DSH_MAX_WRITE_CHARS ?? 12000);
if (Number.isFinite(maxWriteChars) && maxWriteChars > 0 &&
    input.content.length > maxWriteChars) {
  throw new FsError(
    `write content is ${input.content.length} characters; the limit is ` +
    `${maxWriteChars}. Write the file skeleton first, then append the rest ` +
    `with edit or a second write.`,
    "FS_WRITE_TOO_LARGE");
}
```

注意：

- 改的是 `node_modules`，DSH 升级/重装会覆盖；
- 修改前备份 `index.js`；
- 改完重启 DSH；
- 分块第一块本来不完整，不要强制 `</html>` 结尾。

---

## 9. 构建与启动

### 8.1 构建

```powershell
& 'd:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\Launch-VsDevShell.ps1' `
  -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;" +
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;" + $env:PATH
cd I:\llama\kvmem-llama.cpp-multi-backend-framework
cmake --build build-backends/ninfer --target ninfer-serve --parallel 12
```

复制：

```powershell
Copy-Item `
  'I:\llama\kvmem-llama.cpp-multi-backend-framework\build-backends\ninfer\apps\ninfer-serve.exe' `
  'I:\llama\kvmem-llama.cpp-multi-backend-framework\workers\ninfer\ninfer-kvmem-server.exe' `
  -Force
```

### 8.2 推荐 DSH 启动

```bat
--kvmem-budget 40320 --kvmem-gen-reserve 4096 --kvmem-host-mib 12288 ^
--kvmem-keep --kvmem-keep-ratio 20 ^
--kvmem-lrfu ^
--kv-dtype rk8v4 ^
--spec dflash2 --draft-tokens 4 --ngram-draft-tokens 31 ^
--default-reasoning-effort medium --default-thinking-budget 16384
```

注意：当前 fork 不支持 `--reasoning-format deepseek`，不要加。

---

## 10. 测试

### 9.1 日志

```text
KVMEM_PROMPT_PIN system=1/1 tools=135/135 brief=14/14 pages=...
KVMEM_PIN begin=... thinking_end=... frontier=... pinned=... demoted=...
KVMEM_RESELECT logical=... selected=...
```

### 9.2 DSH 30 轮

- 观察 `tools=135/135` 是否完整；
- 统计 `INVALID_ARGS` 数量；
- 对比开启/关闭 `--kvmem-keep` 的 prefix cache 和 prefill 耗时；
- 确认空参数 tool-call 不再出现在 DSH 日志。

### 9.3 LongMemEval

```powershell
cd I:\llama\LongMemEval
.\.venv\Scripts\python.exe .\run_kvmem_memory_180k.py `
  --data .\data\longmemeval_s_cleaned.json --idxs 0,1,2 `
  --target-tokens 192000 --max-server-tokens 258000 --num-questions 20 `
  --write-tokens 1 --answer-tokens 1024 --batch-tokens 12000 `
  --base-url http://127.0.0.1:18200/v1 `
  --out .\out\dsh_keep_192000.json
```

重点确认 128K/192K 不退化。

---

## 11. 回退

### 10.1 代码

当前分支：

```powershell
cd I:\llama\kvmem-llama.cpp-multi-backend-framework
git branch --show-current     # exp/auto-pin
git diff > docs\dsh-integration-20261028.patch
```

回退到 main：

```powershell
git switch main
git restore backends/ninfer/include/ninfer/types.h `
            backends/ninfer/src/models/qwen3_5/program/program_impl.h `
            backends/ninfer/src/models/qwen3_5/program/storage/kvmem_window.cpp `
            backends/ninfer/src/serve/http_server.cpp `
            backends/ninfer/src/serve/openai_chat_response.cpp `
            backends/ninfer/src/serve/serve_options.cpp `
            backends/ninfer/src/serve/translate.cpp
```

或直接：

```powershell
git stash push -u -m dsh-integration
```

### 10.2 启动参数

旧写法仍然兼容：

```bat
--kvmem-pin-answer --kvmem-pin-ratio 20
```

### 10.3 exe 备份

```text
I:\llama\kvmem-backup-20261026-210823\workers\ninfer\ninfer-kvmem-server.exe
```

或回退后重新编译。

---

## 12. 已知限制

1. 本次改动**未编译、未提交**；
2. `--kvmem-keep` 关闭时 system/tools/brief 完全不 pin，prefix cache 可能变差；
3. 空参数过滤只是“丢弃”，不补参数；模型仍可能反复生成空调用；
4. `write` 长度限制尚未加到 DSH；
5. 0.19 RC1 没有这套 `keep` / `translate.cpp` / 空参数过滤逻辑，不能直接套用；
6. `translate.cpp` 的参数顺序只影响 prompt，不改变 JSON Schema 校验。
