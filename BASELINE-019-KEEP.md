# 0.19 基线 + keep 移植记录（2026-11-01）

## 源码状态

```text
backends/ninfer
  base   63428ac (iamwavecut/ninfer-all)
  patches 15 个，来自 kvmem v0.19.1 backends/patches
          (kvmem patch 已包含在 63428ac，reverse check 通过，跳过)
  manual  computed_prompt_tokens 字段（query-memory patch 的一部分）
          cuda-graph-probe patch（单独应用）
  local   keep 移植（见下）

kvmem
  v0.19.1
```

patch 顺序：

```text
ptq1
ptq1-performance
t5-performance
query-memory
nvfp4-a16
nvfp4-int8
nvfp4-kvmem
checkpoint-policy
monitor-events
nvfp4-blackwell-scale
dflash2-gguf-blocks
dflash2-nvfp4
cuda-graph-probe
token-alignment
output-budget
```

## 备份

```text
I:\llama\back\ninfer-fork-g47d43eb-20261101
I:\llama\back\kvmem-fork-20261101
```

## 编译

```text
脚本  C:\Users\zhuan\AppData\Local\Temp\opencode\build_019.bat
源    backends/ninfer
构建  build-backends/ninfer
目标  ninfer-serve
产物  build-backends/ninfer/apps/infer-serve.exe
```

build_019.bat 关键环境：

```bat
call "D:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
set "VCPKG_ROOT=C:\vcpkg"
set "VCPKG_TARGET_TRIPLET=x64-windows"
```

## keep 移植清单

```text
frontend/chat_template.h/.cpp    first_user_region / tools_region
frontend/processor.h/.cpp        字节->token 映射
frontend/prepared_prompt.h       字段
frontend/frontend.cpp            processed/encoded 两条路径填充
include/ninfer/engine_options.h  KvmemOptions::keep
serve/serve_options.cpp          --kvmem-keep
program/program_impl.h           system/tools/task/brief/dynamic_sink
program/storage/kvmem_window.cpp retain_prompt_spans
                                 prepare_window mandatory
                                 rewind_memory_query query_required
                                 restore_memory_history 后重新发布 pins
transactions/commit.cpp          retain_prompt_spans 调用
```

## action query（--kvmem-query-source）

```text
user    默认，query = memory_query（最新 user 内容）
action  query = 上一轮 assistant tool_call 的参数
auto    tool 续接用 action，否则 user
```

触发条件：最后一条消息是 tool，或倒数第二条是 tool（工具结果回传后的追问）。

实现路径：

```text
chat_template   ContentSource::tool_calls_tag
                RenderedChat::action_region
processor       ProcessedInput/EncodedChat::action_region
prepared        PreparedPromptData::action_query
program         memory_query_span 按 query_source 选择
serve           --kvmem-query-source user|action|auto
```

## 验证结果

### 192K 测试口径（固定）

```text
--target-tokens 192000
最终 prompt 204,210，16 轮
不要用 190000：只到 191,458 / 15 轮，mid 覆盖不足，会假性掉题
wrapper: I:\llama\LongMemEval\run-192k.bat
```

### 多轮暗号

```text
turn1 OK
turn2 蓝色鲸鱼
turn3 蓝色鲸鱼（12,929 token，10k filler）
KVMEM_PROMPT_PIN system=15 tools=38 task=1 brief=1 sink=51
```

### 192K 召回（target 192000，final 204210）

```text
keep on   18/20
错误      mid-original、mid-detail-3

对照
  019_aligned（无 keep）  18/20，同样错 mid-original + mid-detail-3
  019rc1                  19/20，mid-original 正确
  47d43eb fork            19/20，mid-detail-3 错
```

keep 没有引入额外退化；mid-original 是 0.19 源码基线与 rc1 发布版的差异。

### decode（8k output，xhigh，draft 4 / ngram 31）

```text
keep on   68.9 tok/s，接受率 47.0%
无 keep   66.6 tok/s（更早一次）
0.19 发布版真实会话 70.15 tok/s
fork g47d43eb      71.4 tok/s
```

## 部署

```text
workers/ninfer/ninfer-kvmem-baseline-keep.exe
workers/ninfer/start-kvmem-ninfer-dflash.bat 已指向该 exe
原 0.19 发布版 exe 保留未覆盖
```

启动参数：

```text
--max-context 262144
--prefill-chunk 256
--kvmem-budget 41600 --kvmem-gen-reserve 4096 --kvmem-host-mib 12288
--kvmem-query-scorer page-softmax --kvmem-lrfu --kvmem-keep
--kv-dtype rk8v4
--spec dflash2 --draft-tokens 4 --ngram-draft-tokens 31
--vision --vision-residency cpu --vision-max-merged 256
--default-reasoning-effort xhigh
```

## 已知问题

```text
1. 192K mid-original 比 019rc1 少 1 题
   需要对比 rc1 的 Query selection / token-alignment 版本

2. 尚未移植 retain_generated_spans（thinking 降权）
   当前请求内 thinking 页不参与 demote
   如果后续出现 thinking 挤占检索预算，需要补
```

## 2026-11-01 更新

### action query

```text
--kvmem-query-source user|action|auto
默认 auto（2026-11-01 起）
```

默认 auto 的验证（不加任何 query-source 参数）：

```text
prompt 56,719，tool 链 + 远处历史
回答 beta.txt -> 蓝色鲸鱼
```

实现：

```text
chat_template  ContentSource::tool_calls_tag
               RenderedChat::action_region
processor      ProcessedInput/EncodedChat::action_region
prepared       PreparedPromptData::action_query
program        memory_query_span 按 query_source 选择
```

触发条件：最后一条消息是 tool，或倒数第二条是 tool。

### 回归

```text
auto 模式 192K target 192000 -> final 204210，18/20
错误 mid-original、mid-detail-3
与 user 模式一致；纯问答回退 user，无污染
```

action query 的收益还需要更严格的 tool 链对照（远处历史 + 模糊问题）。

### 192K 测试口径

```text
必须 --target-tokens 192000
wrapper: I:\llama\LongMemEval\run-192k.bat
190000 只到 191458 / 15 轮，会假性掉题
```

### 部署

```text
workers/ninfer/ninfer-kvmem-baseline-keep.exe
  已更新为 action query build
```
