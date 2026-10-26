# KVMem L1 retention, structural prompt pins, and masked-draft checkpoint fixes (2026-10-26)

本文档说明本次本地提交的内容、验证结果和回退方式。

- 本地标签：`kvmem-l1-20261026`（基础 L1 + checkpoint 修复）
- 本地标签：`kvmem-prompt-pin-20261027`（结构化 tools / brief pin）
- 改动前备份：`I:\llama\kvmem-backup-20261026-210823`（6.19 GiB，含 `.git` / build / workers）
- 远程：**未推送**，仅本地提交

---

## 1. 本次改动

### 1.1 四个隐性 bug

| # | 文件 | 问题 | 修复 |
|---|---|---|---|
| 1 | `storage/disk_tier.cpp` | 磁盘恢复只设置 `dflash_context_frontier`，没有设置 `dflash_context_floor`，复用序列槽位时可能留下过期的 floor | 恢复时按 `max(recent_blocks * 64, masked_draft_ring_capacity())` 重设 floor 并 clamp |
| 2 | `storage/context.cpp`、`storage/kvmem_window.cpp` | DFlash2 的 ring append 延迟到下一轮，checkpoint 捕获时 `dflash_context_frontier` 可能落后，导致 `capture_memory_checkpoint` 直接跳过（或恢复出不完整的 ring） | 新增 `sync_masked_draft_frontier()`：捕获前补齐 gap；prefill / fork / 前后 frontier 不一致时跳过，交给原校验拒绝 |
| 3 | `storage/kvmem_window.cpp` | reselect 时 draft floor 用 `end - ring`，比 `begin - ring` 多掩掉最新 `width` 个 ring token | 改为 `begin - ring`（`begin` 是已提交 frontier） |
| 4 | `prefill.cpp` | DFlash2 主机侧 `token_counts` 只加 `accepted` 个 token，漏掉 terminal token；dense 路径加的是 `accepted + 1` | 改为 `min(accepted + 1, verify_drafts + 1)` |

第 2 项在测试中可以观察到：

```text
KVMEM_DFLASH_SYNC frontier=5279 gap=5
KVMEM_PREFIX_CHECKPOINT frontier=5279
KVMEM_ENDPOINT frontier=5301
```

### 1.2 L1：thinking-aware 软保留

新增 `Program::retain_generated_spans(sequence, generated_begin, thinking_end)`，engine 在 `finish()` 前调用。

分页规则（64-token page）：

```text
[generated_begin, thinking_end)      -> demoted_pages（软降权，最后才检索）
[thinking_end, execution_frontier)   -> pinned_pages（mandatory）
```

- 混合页（thinking 尾部 + answer 头部）优先 pin，不拆页；
- 请求开头与 prompt 混合的页不 demote；
- pinned 上限为 B 的 `pin_answer_percent`（默认 20%，范围 1–50），超出保最新；
- pinned 加入 `prepare_window()` 的 mandatory；
- demoted 的检索分数置 `-1e30`，fallback 选择也先跳过；
- `pinned_pages` / `demoted_pages` 随 `KvmemWindowState` 和 `KvmemPrefixCheckpoint` 一起保存/恢复，跨请求 reuse 和 checkpoint restore 都保留。

这是**软 pin**：thinking 页仍在 Host archive 中，只有在非 thinking 页不够时才会被选中；不会物理删除，也不影响 DFlash2 的 target 验证正确性。

### 1.3 结构化 prompt pin（2026-10-27）

在 L1 的页级机制上，增加两个**内置**的 prompt 保留类别，不新增命令行参数：

| 类别 | 预算 | 页选择 |
|---|---:|---|
| tools schema | B 的 **8%** | 从 tools 区间开头取整页；超出时保留前面的定义 |
| 第一条 user brief | B 的 **4%** | brief 较短时全保留；超过预算时保留第一页 + 末尾若干页（长粘贴内容通常任务在末尾） |

- `tools_region` / `first_user_region` 由 chat template 和 tokenizer 逐级传到 `PreparedPromptData`；
- 在 lane 绑定后的 commit 阶段调用 `ProgramImpl::retain_prompt_spans()`，写入独立的 `tool_pages` / `brief_pages`；
- `prepare_window()` 的 mandatory 分配顺序：

```text
sink > tools > brief > recent > answer > 普通检索 > demoted thinking
```

- tools/brief 页不会被 answer 的“保最新”上限挤掉；
- 混合页整页 pin；
- tools/brief 页从 `demoted_pages` 中移除；
- 随 checkpoint 保存/恢复；恢复时如果当前请求已经带 prompt spans，优先保留当前请求的，不被旧 checkpoint 覆盖；
- 如果模板没有暴露 tools 区间，退化为原来的位置 sink。

### 1.4 新增参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--kvmem-pin-answer` | off | 开启 L1：pin 非 thinking 尾部，demote 完整 thinking 页 |
| `--kvmem-pin-ratio N` | 20 | pinned 页上限，占 B 的百分比，1–50 |

只加 `--kvmem-pin-answer` 即可；`--kvmem-pin-ratio` 可省略。

验证日志（需 `--kvmem-verify-transfers`）：

```text
KVMEM_PIN begin=5046 thinking_end=5279 frontier=5301 pinned=1 demoted=3
KVMEM_RESELECT logical=9216 ... selected=...,82,...
```

---

## 2. 测试结果

构建：Release，`CMAKE_CUDA_ARCHITECTURES=120a`，产物 `apps/ninfer-serve.exe` 550,006,784 字节，已复制到 `workers/ninfer/ninfer-kvmem-server.exe`。

### 2.1 小窗口强制驱逐（B=4096 / R=4096 / W=8192）

| 请求 | 结果 |
|---|---|
| Req1 | prompt 5,046 / reasoning 233 / frontier 5,301；`KVMEM_PIN pinned=1 demoted=3`；`KVMEM_DFLASH_SYNC gap=5` |
| Req2 | prompt 11,636 / cached 5,301；`KVMEM_RESELECT logical=9216`；selected 含 page **82**（上一轮 answer），不含 79/80/81（thinking）；decode 67.7 tok/s，接受率 50.3% |

### 2.2 真实窗口（B=32,448 / R=16,384 / W=48,832 / rk8v4）

| 测试 | 结果 |
|---|---|
| 短请求 | decode 65.4 tok/s，接受率 48.1% |
| 25K prompt | prefill 845.1 tok/s，decode 77.3 tok/s，接受率 84.6% |
| 55K prompt | `KVMEM_RESELECT logical=48640 selected=507`；decode 51.3 tok/s，接受率 50.0%，无崩溃/无错误 |
| frequency/presence penalty | 正常返回，无异常 |

### 2.3 最终 smoke

```text
DFLASH2 FINAL OK
decode 203.6 tok/s（短请求）
dflash2 accepted 4/7 (57.1%)
```

### 2.4 结构化 prompt pin

| 测试 | 结果 |
|---|---|
| 单工具短请求 | `KVMEM_PROMPT_PIN tools=2/40 brief=9/20 pages=15`；工具调用 `write` 参数完整 `{"file_path":"ok.txt","content":"OK"}` |
| 55K prompt + 单工具 | `KVMEM_PROMPT_PIN tools=2/40 brief=20/20 pages=871`；`KVMEM_RESELECT logical=48640 selected=507`，selected 以 `0,1` 开头（tools 页） |
| 长 prompt DFlash2 | prefill 431.8 tok/s；decode 30–33 tok/s（输出很短）；无错误 |
| 短 HTML 请求 | decode 96.0 tok/s，接受率 93.5% |
| 工具调用 | decode 80.8–92.8 tok/s，接受率 87.5–92.9%，required 字段完整 |

结论：tools span 能正确提取；结构化页进入 mandatory；不再出现“工具调用缺 `file_path`/`content`”的 schema 漂移（本次样本内）。

测试日志：

- `C:\Users\zhuan\AppData\Local\Temp\opencode\kvmem_structural_server.err.log`
- `C:\Users\zhuan\AppData\Local\Temp\opencode\kvmem_tailpin_server.err.log`
- `C:\Users\zhuan\AppData\Local\Temp\opencode\kvmem_final5_server.err.log`

---

## 3. 当前运行状态

- 18200 使用新二进制，配置：q4km / B=32,448 / R=16,384 / recent 1,024 / rk8v4 / DFlash2；
- **L1 已启用**：`--kvmem-pin-answer --kvmem-pin-ratio 20`；
- 结构化 tools/brief pin **内置生效**，无额外参数；
- 当前进程未加 `--kvmem-verify-transfers`，需要看 `KVMEM_PROMPT_PIN` 日志时再临时加。

关闭 L1（保留结构化 pin）：

```bat
去掉 --kvmem-pin-answer --kvmem-pin-ratio 20
```

---

## 4. 回退

### 4.1 只关闭 L1（不重编译）

从启动参数去掉：

```bat
--kvmem-pin-answer --kvmem-pin-ratio 20
```

其他行为不变。

### 4.2 回退本次提交

只回退结构化 prompt pin、保留基础 L1：

```powershell
cd I:\llama\kvmem-llama.cpp-multi-backend-framework
git reset --hard kvmem-l1-20261026
```

完整回退（连 L1 一起）：

```powershell
git reset --hard kvmem-l1-20261026^
```

或保留历史：

```powershell
git revert <commit>
```

### 4.3 只恢复二进制

```powershell
Copy-Item `
  'I:\llama\kvmem-backup-20261026-210823\workers\ninfer\ninfer-kvmem-server.exe' `
  'I:\llama\kvmem-llama.cpp-multi-backend-framework\workers\ninfer\ninfer-kvmem-server.exe' -Force
```

### 4.4 整目录恢复

```powershell
robocopy `
  'I:\llama\kvmem-backup-20261026-210823' `
  'I:\llama\kvmem-llama.cpp-multi-backend-framework' `
  /E /COPY:DAT /DCOPY:DAT /R:1 /W:1
```

### 4.5 重新编译

```powershell
cmd /c 'call "D:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul && cd /d I:\llama\kvmem-llama.cpp-multi-backend-framework\build-backends\ninfer && ninja ninfer-serve.exe'
Copy-Item `
  'I:\llama\kvmem-llama.cpp-multi-backend-framework\build-backends\ninfer\apps\ninfer-serve.exe' `
  'I:\llama\kvmem-llama.cpp-multi-backend-framework\workers\ninfer\ninfer-kvmem-server.exe' -Force
```

---

## 5. 未纳入本次提交

- `workers/ninfer/start-kvmem-ninfer-dflash.bat`
- `workers/ninfer/start-kvmem-ninfer.bat`
- `docs/kvmem-update-20261026.html`

这两个 bat 与 HTML PPT 是本次之前就存在的工作区改动，保持原样。

---

## 6. 已知限制

- 磁盘 tier 的 floor 修复只做了编译验证；本次测试未启用 `--kvmem-disk-path`。
- L1 是页粒度（64 token），thinking/answer 同页时整页 pin。
- 结构化 tools pin 上限是 B 的 8%；工具 schema 超过该预算时只保留前面的定义，剩余部分仍靠检索。38 工具以上的 harness 建议压缩 schema 或提高 B。
- brief 超过 4% B 时保留第一页 + 末尾页；长粘贴内容中间的段落不保证常驻。
- L1 只影响 reselect / history reuse；单轮短请求不会触发。
- 画质 A/B（同 prompt、开/关 pin、多轮 HTML 动画）尚未做；本次验证的是机制正确性和 DFlash2 运行正常。
- pinned 页若在 checkpoint 时不在 resident view，恢复后要等下一次 reselect 才召回；上一轮答案通常仍在 recent 中，测试中已覆盖。
