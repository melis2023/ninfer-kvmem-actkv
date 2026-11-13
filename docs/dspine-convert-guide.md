# DSpine GGUF → ninfer 转换说明

把 [aj9o9/Qwen3.8-27B-DSpine-b16-GGUF](https://hf-mirror.com/aj9o9/Qwen3.8-27B-DSpine-b16-GGUF/tree/main)
里的 DSpine 草稿合并进一份 Qwen3.8-27B 的 ninfer artifact。

*English abstract: how to merge the published DSpine drafter GGUF with a Qwen3.8-27B
target GGUF into one `.ninfer` artifact (GUI or CLI), verify it, and serve it with
`--spec dflash`.*

该仓库提供的是**外挂草稿**（`Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf`，约 1.17 GiB），
不能单独运行；需要和一个 **Qwen3.8-27B 目标 GGUF** 合并，产物里会同时含
`text`、可选的 `mtp` 和 `dflash` 三个组件。

## 1. 准备

| 需要 | 说明 |
| --- | --- |
| 目标 GGUF | 任意 Qwen3.8-27B 量化版；需要 MTP 就选带 `-mtp` 的（例如 `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`，11.29 GiB） |
| 草稿 GGUF | `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf`（HF 仓库） |
| 转换器 | `gguf转ninfer工具`（GUI，勾选「DSpine 草稿」）或本仓库 `backends/ninfer/tools` |
| 环境 | Python 3.10+、`numpy`、`torch`（CPU 版即可）、磁盘 ≥ 15 GB |

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
huggingface-cli download aj9o9/Qwen3.8-27B-DSpine-b16-GGUF `
  Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf --local-dir I:\model
```

## 2. 图形界面（推荐）

1. 双击 `NInfer-Tool.bat`；
2. 勾选 **「DSpine 草稿」**（DFlash2 下面一行），选择
   `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf`；
3. 主模型选 `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`（自带 MTP 头会自动带入）；
4. 输出名自动带 `-dspine` 后缀，点「开始转换」。

实测：**约 60 秒**（CPU）、**12.9 GiB（13.8 GB）**、898 个对象，产物可直接被
`ninfer-serve` 加载。

## 3. 命令行

```powershell
cd <转换器根目录>          # GUI 版: ...\ninfer-convert\dspine ；本仓库: backends\ninfer\tools

python -m tools.convert `
  --model "<metadata 目录>" `
  --recipe qwen3_5_dense_gguf `
  --source "gguf=<目标.gguf>" `
  --source "dflash=<草稿.gguf>" `
  --components text,mtp,dflash `
  --override tools/convert/recipes/dspine_gguf_q4.py `
  --device cpu --rows-per-chunk 512 `
  --name <实例名> `
  --out "<产物.ninfer>"
```

`--model` 是 HF metadata 目录（GUI 版为 `conversion-input/metadata`，本仓库即
`backends/ninfer/tools/convert` 同级的 metadata 输入）。

四个关键点，错一个就转不出可用产物：

- `--source dflash=`（**不是** `dflash2`）：DSpine 草稿走 `dflash` 组件；
- `--components` 必须含 `dflash`，要 MTP 再加 `mtp`；
- `--override` 指向 `dspine_gguf_q4.py`：草稿的 Q4_K 块**按原字节直通**，只有视觉塔压到
  Q4、selected-prefix bridge 保持 BF16（bridge 要按行 gather codebook）；
- base recipe 用 `qwen3_5_dense_gguf`（旧的 `qwen3_8_27b_gguf` 转换器没有 dspine 支持）。

## 4. 校验

```powershell
python tools\dspine_audit.py --gguf "<草稿.gguf>" --artifact "<产物.ninfer>"
python tools\dspine_smoke.py --artifact "<产物.ninfer>"
```

`dspine_audit.py` 比对 73 个草稿张量、tap 列表与 bridge 复合权重；`dspine_smoke.py` 做
形状与 dtype 冒烟。

## 5. 运行

```powershell
ninfer-serve.exe "<产物.ninfer>" --port 18200 `
  --max-context 256000 --prefill-chunk 512 `
  --kvmem-budget 47104 --kvmem-gen-reserve 2048 --kvmem-host-mib 12288 `
  --kv-dtype nvfp4 --kvmem-query-scorer meanq-cosine --kvmem-lrfu `
  --spec dflash --draft-tokens 4 --full-draft-head --ngram-draft-tokens 0
```

- `--spec dflash` 就是 DSpine 后端；与 `--spec mtp`、`--spec dflash2` 互斥；
- 16 GB 卡上 `--draft-tokens 4` 最优（k=6 短上下文更快，但 100K 以上慢约 25%）；
- `--full-draft-head` 必需：DSpine 的 16×16 lattice 候选取自全词表 top-16；
- 草稿常驻约 1.2 GiB，启动器要留出 ≥ 640 MiB 的空闲显存（引擎日志会 WARN）。

## 6. 常见报错

| 报错 | 原因 / 解法 |
| --- | --- |
| `DSpine requires the full draft proposal head` | 产物必须用全词表 head，启动别加 `--lm-head-draft` |
| `dspine_bridge: invalid context_k` | 草案 KV 目前只支持 BF16，别加 `--draft-kv-dtype int8` |
| `the DSpine GGUF recipe requires the dflash component` | 转换时漏了 `--components …,dflash` 或 `--source dflash=` |
| `No module named 'dspine_gguf'` / 无 `dspine_gguf.py` | 用了旧转换器；用 `dspine` 子目录（GUI 勾选后自动切换）或更新到含本说明的提交 |
| 运行掉速 / 抖动 | 看启动日志的显存告警：空闲 VRAM < 640 MiB 会触发 WDDM 换页 |

## 7. 参考数据（RTX 5060 Ti 16 GB，k=4，nvfp4 KV）

| prompt | DSpine decode | 接受率 |
| --- | --- | --- |
| 25 token | 89–94 tok/s | 76–77% |
| 15,325 token | 70–84 tok/s | 64–71% |
| 117,771 token | 51–58 tok/s | 46% |
| 202,604 token | 66–70 tok/s | 62% |

同 prompt 下与 DFlash2 对比：15K +36%、118K +26%、203K +28%，差距全部来自接受率。
