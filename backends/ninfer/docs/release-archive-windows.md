# NInfer for Windows — release archive

This is the README that ships **inside** the Windows release archive, with command executables at its root and CUDA libraries in backend subdirectories. If you are reading it in a checkout instead, the launchers and downloaders it names live
under `scripts/`, and
[docs/build-windows.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/build-windows.md)
is the fuller guide. Links here are absolute on purpose: the archive ships no `docs/` directory.

Check `BUILD-INFO.json` for the exact source commit, CUDA version and device images.
`VERSION` records the source snapshot's version field. Matching `RELEASE_NOTES_*.md` are included when present.

Download **all** `.7z.*` volumes for one package into the same directory. Use 7-Zip to extract
`.7z.001`. The `cuda12.9-13.4` package contains both backends. Do not flatten or merge their subdirectories. The outer `SHA256SUMS-*.txt`
checks the downloaded volumes. The inner `SHA256SUMS.txt` checks the extracted files.

## Requirements

- Windows 11 x64
- an NVIDIA GPU covered by the device images in `BUILD-INFO.json`, with a driver compatible with
  the package's CUDA version
  - CUDA 12.9 CI package: Volta, Turing, Ampere (including A100), Ada
  - CUDA 13.4 CI package: Turing, Ampere (including A100), Ada, consumer Blackwell
  - local packages carry the architectures selected by their builder
- Microsoft Visual C++ 2022 runtime

Everything else is in the archive: the FFmpeg and libcurl DLLs, and NVIDIA's cuBLAS runtime
(`cublas64_12.dll` and `cublasLt64_12.dll`, or their CUDA 13 counterparts). You do **not** need the CUDA Toolkit installed.
Packages built with `ggml_kernel_dll: true` in `BUILD-INFO.json` also require the included
`ninfer_ggml_quants.dll`. Keep it in its backend directory; it contains NInfer's GGML CUDA kernels
for the package's full architecture set.

**Licensing.** NInfer is Apache-2.0 (`LICENSE`). The two cuBLAS DLLs are NVIDIA's, not part of NInfer:
they are redistributed under the terms of the NVIDIA CUDA Toolkit EULA (`backends/cuda12/NVIDIA-CUDA-EULA.txt` or `backends/cuda13/NVIDIA-CUDA-EULA.txt`,
Attachment A), for use only with the NInfer executables in this folder, and they are not covered by the
Apache licence.

Model artifacts are **not** included — they are 17–21 GB each. The downloaders below fetch them.

## Backend selection

The universal package selects CUDA 12 for Volta and CUDA 13 for consumer Blackwell. On overlapping
architectures it prefers CUDA 13 when the driver supports it, otherwise CUDA 12. Use
`--cuda-backend auto|12|13` to override the choice. The minimum Driver API family is 12.0 or 13.0,
respectively. Admission does not establish GPU correctness on every driver release.
All devices in one process must fit the same backend. Keep `backends/` and `help/` beside the EXEs.
`--help` and `--version` run without an NVIDIA driver. Each command loads the same private runtime
DLL for its selected CUDA family. It does not load both families into one process.

## Quick start

From the directory you unpacked, two commands (or double-click either file and pick a model):

```powershell
.\download-model.bat qwen38-27b        # ~19 GB, resumable, verifies size and SHA256; the DFlash2 bundle, it also carries the MTP weights
.\run.bat qwen38-27b                   # serves on http://127.0.0.1:8080/v1
```

The downloader writes into `models\` beside these files, which is where the launcher looks.
`NINFER_MODEL_DIR` moves both: set it and the downloader puts artifacts there and the launcher looks
there. To point the launcher at one file somewhere else, give it `NINFER_MODEL`.

The launcher defaults were tuned on an RTX 3090 with 24 GB. Other cards may need smaller context
and concurrency settings. The default Qwen3.8 profile uses Qwen3.8-27B, one user, 172,032 tokens, DFlash2 speculation
plus the draft head (fastest), `rk4v4` KV, cuBLAS prefill, and vision in overlay residency.

For the longest context instead, `set NINFER_SPEC=mtp && .\run.bat qwen38-27b` runs the MTP profile at
the full 262,144 tokens shared by two lanes, with a smaller prefill chunk and `--lm-head-q6`: slower
decode, more context.

`run.bat qwen38-27b int8` and `run.bat qwen38-27b c8` are the older INT8 profiles — one user at
65,536 tokens, and eight concurrent users at 8,192 each.

For the Qwen3.6-35B-A3B MoE instead, which serves more lanes:

```powershell
.\download-model.bat qwen36-35b-a3b    # ~21 GB, resumable, verifies size and SHA256
.\run.bat qwen36-35b-a3b               # two lanes sharing 262,144 tokens, MTP3 + draft head, vision
```

The endpoint is OpenAI-compatible, so anything that speaks `/v1/chat/completions` works. Leave the
API key blank.

## What is in the archive

| file | what it is |
|---|---|
| `ninfer-serve.exe` | the server: OpenAI- and Anthropic-compatible HTTP APIs |
| `ninfer.exe` | one-shot CLI generation, for smoke tests and scripting |
| `ninfer_bench.exe` | throughput benchmark against the public Engine route |
| `backends/cuda*/ninfer_ggml_quants.dll` | GGML CUDA kernels when `ggml_kernel_dll` is enabled |
| `backends/cuda*/ninfer-runtime.dll` | shared Engine and command implementations |
| `help/*.txt` | command help without loading CUDA |
| `ninfer-calibrate.exe`, `ninfer-perplexity.exe`, `ninfer-ngram-profile.exe` | calibration, scoring and n-gram tools |
| `download-model.bat` | pinned, resumable artifact downloads with verification |
| `run.bat` | serving profiles: `run.bat <model> [profile]`, or double-click it to pick a model |
| `backends/cuda*/*.dll` | the FFmpeg and libcurl runtime dependencies, and NVIDIA's cuBLAS for the selected CUDA version |
| `backends/cuda*/NVIDIA-CUDA-EULA.txt` | the licence under which the two cuBLAS DLLs are redistributed |
| `SHA256SUMS.txt` | checksums for the package files, including subdirectories |
| `BUILD-INFO.json` | source, Toolkit, device images and validation scope |
| `LICENSE-*.txt` | dependency notices from vcpkg |

## Overrides

Nothing here needs editing. **Every** profile reads `NINFER_MODEL`, `NINFER_MODEL_DIR`,
`NINFER_SERVER`, `NINFER_HOST`, `NINFER_PORT` and `NINFER_CHAT_TEMPLATE` (a local Jinja file,
passed to `--chat-template`, overriding the artifact's built-in template). The default (`tuned`)
profiles read more:

| profile | also reads |
|---|---|
| `run.bat qwen38-27b` | `NINFER_CONTEXT`, `NINFER_CONCURRENCY`, `NINFER_KV_DTYPE`, `NINFER_SPEC` (`dflash2`, `mtp`, `none`), `NINFER_DRAFT_TOKENS`, `NINFER_PREFILL_CHUNK`, `NINFER_VISION`, `NINFER_VISION_RESIDENCY`, `NINFER_HOST_STATE_SLOTS`, `NINFER_FALLBACK` (`off` turns the automatic step-down off) |
| `run.bat qwen36-35b-a3b` | the same, except `NINFER_SPEC` is `mtp` or `none` |
| `run.bat qwen38-27b int8`, `run.bat qwen38-27b c8` | nothing further; every serving flag is fixed |
| `download-model.bat <model>` | `NINFER_MODEL_DIR` |

> `NINFER_HOST=0.0.0.0` exposes the server to your network **unauthenticated**. The launchers bind
> `127.0.0.1` for that reason. If you set it, put something in front of it.

## If startup refuses

**The default profile handles this for you.** If `run.bat` is refused at startup for lack of GPU
memory, it steps down by itself -- an eighth of the context at a time, up to five times, and from
the second step with a 2048 prefill chunk and fewer host state slots -- says what it did, and starts. On a desktop that was
holding 2.8 GiB of the card, `run.bat qwen38-27b` stepped down to 81,920 tokens and
served a request, at about 12 seconds per refused attempt. It only does this for the defaults: a
`NINFER_CONTEXT`, `NINFER_PREFILL_CHUNK` or `NINFER_HOST_STATE_SLOTS` you set is honoured as given, and
`NINFER_FALLBACK=off` turns it off.

The rest of this section is for those cases. The message names the numbers. A 24 GB card running a
desktop has roughly 1.5 GiB less to work with than a headless one, so the largest profiles do not fit
alongside a desktop:

```
requested Engine runtime reservation requires 2864526592 bytes,
but only 2375691264 bytes are available for runtime capacity
```

Drop a context rung first: set `NINFER_CONTEXT` to the next rung below the profile's default; the
rungs are listed in the `run.bat` header, measured on this card — for
`run.bat qwen38-27b` (default 172,032 with DFlash2, 262,144 with `NINFER_SPEC=mtp`) that is
163840, then 131072, then 98304. Speculation is the next lever (`NINFER_SPEC=none`), worth about
992 MiB on the 35B-A3B at the cost of decode speed.
Drop vision last: in overlay residency it costs almost nothing resident.

To size a profile before running it, open
[docs/config-calculator.html](https://github.com/iamwavecut/ninfer-all/blob/master/docs/config-calculator.html)
from the repository — one self-contained file, no network needed, every constant in it measured on
an RTX 3090. It is not in this archive.

## Full documentation

<https://github.com/iamwavecut/ninfer-all>
