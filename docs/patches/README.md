# ninfer 子模块补丁

`backends/ninfer` 指向本地 commit `e530d1ed`。上游 `iamwavecut/ninfer-all.git` 对当前账号只读（push 返回 403），所以这个 commit 不在任何远程上。新建 clone 后需要按下面的步骤恢复。

## 恢复步骤

```bash
git submodule update --init backends/ninfer
cd backends/ninfer
git checkout 8319e8f5          # 所有补丁的基线
for p in ../../docs/patches/000*.patch; do git am "$p"; done
```

应用后的 HEAD 应当是 `e530d1ed`，`git log --oneline -5` 能看到 5 个补丁提交。

## 补丁一览

| 文件 | 内容 | 来源 |
|---|---|---|
| `0001-feat-qwen3_5-DFlash2-KVMem-window-recall-with-merged.patch` | DFlash2 + KVMem 窗口召回，合并页run 传输 | 本地 |
| `0002-perf-kvmem-start-reselection-at-80-of-the-window.patch` | 重选起点提前到窗口 80% | 本地 |
| `0003-perf-kvmem-score-token-only-prompts-on-their-tail.patch` | `/v1/completions` 原始文本取尾部打分 | 本地 |
| `0004-perf-kvmem-restore-only-changed-pages.patch` | 差异 restore，只搬变化的页 | 本地 |
| `0005-fix-serve-tool-request-sampling-without-presence-penalty.patch` | 带 tools 的请求 `presence_penalty` 归零 | 上游 `bff41bea` 的 cherry-pick |
| `0006-feat-ops-sm70-and-sm75-build-support-on-the-compatib.patch` | sm70/sm75 的 31 个实现文件与 device profile | 上游 `08fd7019` 的最小移植 |
| `0007-feat-sm70-sm75-wire-the-compatibility-path-into-mma-.patch` | mma/memory 接入、codec 与 QType 依赖 | 同上 |

## 已验证的基线

| 场景 | 配置 | 结果 |
|---|---|---|
| 256K NIAH（33 针） | budget 36864 / reserve 16384 / chunk 256 / int8 / MTP draft 4 | 33/33 strict、14/14 cross-tool、80 工具 |
| 256K real（33 针，生产 system prompt + 10 工具） | 同上 | 33/33 strict、14/14 cross-tool、240 工具 |
| LongMemEval | 128K / 192K / 240K | 20/20、18/20、16/20 |

## 注意

- 补丁 0001 体积较大（1.1 MB），应用耗时约 10 秒。
- 0005 与 0004 无文件重叠，顺序无关；其余按编号顺序应用。
- 若上游后续改动了 `src/serve/translate.cpp` 或 `src/models/qwen3_5/frontend/chat_template.cpp`，0005 可能需要手工解冲突。
