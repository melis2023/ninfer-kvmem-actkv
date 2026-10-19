# KVMem + ActKV · ninfer 后端

基于 [kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) 与 **ActKV**（[arXiv:2609.31395](https://arxiv.org/pdf/2609.31395)，中国科学技术大学提出）技术开发：

- **KVMem**：把长上下文稀疏化成固定窗口（`budget` 选取历史块 B + `gen_reserve` 追加工作区 R），Host/磁盘归档历史 KV，用一个固定大小的注意力窗口换来 200K/262K 的可用上下文。
- **ActKV**：面向行动的检索保持策略（LRFU 历史排序等），提升长会话中关键块的召回率。

**QQ 交流群：1040777853**

本项目仅供研究学习，稳定使用可前往kvmem原作者。

## 2026-10-18 更新：DFlash2 × KVMem 修复

ninfer 后端的 `--spec dflash2` 在 KVMem 长上下文下现已正常工作。共修复五个问题，
完整根因分析与验证矩阵见 [docs/dflash2-kvmem-fixes-20261018.md](docs/dflash2-kvmem-fixes-20261018.md)：

1. **envelope 短路（主因）**：草稿滑窗注意力的 `max_context` 被设成环容量 2048 并固化进
   CUDA graph，内核守卫让绝对位置超过 2048 的草稿轮全部静默输出零——上下文一超过 ~2K
   token 接受率就归零，与内容无关。
2. **检索缝合污染环**：KVMem 重选换块后环里残留陈旧特征；新增 `dflash_context_floor`
   + 逐行 `min_positions` 掩码，让 2048 滑窗只读尾部连续段。
3. **dflash 轮缺失 `prepare_window`**：B+R 窗口填满后崩溃
   （`KV coverage exceeds active entitlement`，tokens=53250 vs 53248）。
4. **state-slot fork 跟随**：环 lane 跟随 StateImage 槽位变化。
5. **draft-bypass 诊断**：`KVMEM_DFLASH_BYPASS` 记录连续空手神经轮。
6. dflash2模型推荐，Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored-MTP，或者其他低于11G容量，精度格式推荐rk8v4，或者nvfp4。
7. bonsai2模型推荐，https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3/tree/main
已经内置dflash2，无需再次合并


### 256K 召回复测（2026-10-18，RTX 5060 Ti，dflash2 草稿 3，nvfp4，15 轮 / 80 次工具调用）

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 同轮召回 | 33/33 | **33/33** |
| 终轮召回（严格 / 并集） | 33/33 | **33/33** |
| 跨工具召回（send_alert） | 14/14 | **14/14** |
| 终轮召回 decode | ~72.6 tok/s | **106.0–135.8 tok/s** |
| 探针 decode | ~30–40 tok/s | **58.7–98.5 tok/s** |
| 131K decode（greedy，缝合窗口） | 71.4 tok/s | **120.3 tok/s** |

召回质量零损失，decode 最高提升 ~87%。测试驱动为仓库根目录
[`niah33_262k.py`](niah33_262k.py)，结果写入 `niah33-262k-metrics.json`。

## 2026-10-19 更新：召回传输优化（2 倍）

KVMem 每次窗口重选要搬约 2.2 GB，实测只有 3.8–4.5 GB/s，而同一张卡的链路能跑 14.1 GB/s。
根因是 `record_groups` 的语义用错：合并条件要求相邻记录属于同一分配区，调用方却传了块 id
（sink / recent / topk 混排，天然不连续），于是 576 页被拆成 36,864 次单页 26 KiB 拷贝，
耗时全花在驱动提交上，GPU 拷贝引擎几乎空闲（`sync_us` 仅 5 ms）。

改为传入空 groups 后，连续页合并为每 plane 一次 2D 拷贝：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| DMA 带宽 | 4.0 GB/s | **14.1 GB/s** |
| 单次重选传输（256K 均值） | 515 ms | **256 ms** |
| 256K 24 次重选传输合计 | 12.37 s | **6.14 s** |
| 256K prefill | 340.8 tok/s | **343.4 tok/s** |
| 256K 端到端 | 729.8 s | **724.2 s** |

输出逐位不变（256K 接受率 112/343，64K 124/296），`--kvmem-verify-transfers` 回放
`payload_exact=1 state_exact=1`。传输只占端到端 1.7%，所以整体提升有限；
完整带宽基准、24 次逐次对比与剩余优化空间见
[docs/kvmem-recall-opt-20261019.md](docs/kvmem-recall-opt-20261019.md)。

> 子模块 `backends/ninfer` 的上游是第三方仓库（`iamwavecut/ninfer-all.git`），本次改动
> 没有推送权限。完整改动已作为
> [`docs/patches/0001-feat-qwen3_5-DFlash2-KVMem-window-recall-with-merged.patch`](docs/patches/0001-feat-qwen3_5-DFlash2-KVMem-window-recall-with-merged.patch)
> 附带（183 个文件）；在 `backends/ninfer` 里 `git am <该文件>` 即可复现。

## GGUF → .ninfer 模型转换

转换工具可在Releases中下载（Python 3.11 + PyTorch/NumPy）。
`qwen3_8_27b_gguf` 配方把 Qwen3.8-27B GGUF **按自身块格式直接转换**（不做反量化），
可选加入 MTP 头、DFlash2 草稿与视觉投影器。在 `backends/ninfer` 目录下运行：

```bash
python3 -m tools.convert \
  --model /path/to/conversion-input/metadata \
  --recipe qwen3_8_27b_gguf \
  --source gguf=Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored-MTP.gguf \
  --source dflash2=Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --source vision=mmproj-Qwen3.8-27B-Q8_0.gguf \
  --components text,vision,mtp,dflash2 \
  --name qwen3.8-27b \
  --out model.ninfer
```

- `--model` 只需要基础检查点的配置与分词器（准备转换输入时生成的 `metadata` 目录）。
- `dflash2` 源支持官方 DFlash2 GGUF 或其 safetensors 目录。
- 不需要的组件可省略（例如纯文本去掉 `vision`）；`mtp` 组件要求 `-mtp` GGUF
  （`blk.64.nextn.*`）。
- CPU 转换追加 `--device cpu`。

## 快速开始

1. 下载 `.ninfer` 模型（或按上文自行转换）。
2. 双击 `workers/ninfer/start-kvmem-ninfer-dflash.bat`（DFlash2）或
   `start-kvmem-ninfer.bat`（MTP）。
   - 模型路径、启动目录与参数都在 bat 文件里，直接编辑即可。
   - 显存 16 GiB 推荐：`--max-context 262144 --kvmem-budget 36864
     --kvmem-gen-reserve 16384 --kvmem-host-mib 12288`。
3. 服务地址 `http://127.0.0.1:18200`，OpenAI 兼容 API。


## 文档

- [DFlash2 × KVMem 修复记录（2026-10-18）](docs/dflash2-kvmem-fixes-20261018.md)
- [召回传输优化：带宽基准、256K A/B 与逐次对比（2026-10-19）](docs/kvmem-recall-opt-20261019.md)
- [后端能力与启动说明](docs/multi-backend.md)
- [仓库结构与构建](docs/multi-backend-repository.md)
- [后端差异](docs/multi-backend-differences.md)

