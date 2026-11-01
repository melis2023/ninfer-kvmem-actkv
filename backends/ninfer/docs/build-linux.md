# Build NInfer on Linux

This guide runs NInfer on Linux from the published container image, or builds it natively.
`CMAKE_CUDA_ARCHITECTURES` selects one target per build directory:

| value | GPUs | toolkit |
|---|---|---|
| `86` (the default) | RTX 30 series; an `86` build also runs on the RTX 40 series | CUDA 12.8 or newer |
| `89` | RTX 40 series only | CUDA 12.8 or newer |
| `120a` | RTX 50 series and RTX PRO 6000 Blackwell | CUDA 13.1 or newer (configure refuses 12.8 and 12.9, which miscompile `sm_120a` kernels) |
| `80` | A100 class (GA100), an unmeasured compatibility target | CUDA 12.8 or newer |
| `70` | Tesla V100 (Volta) | CUDA 12.8 or 12.9 (CUDA 13 no longer compiles `sm_70`) |
| `75` | TITAN RTX and RTX 20 series (Turing), through the [Turing tier](maintainer/build-system.md#turing-tier-75) | CUDA 12.8 or newer |

An `89` or `120a` build does not run on an RTX 3090, only a `70` build runs on a V100, and only
a `75` build on a Turing card. See [Turing](#turing-sm_75) and [Tesla V100](#tesla-v100-sm_70) for
what those builds change.

## Container image

The release pipeline packages Debian 13 images from prebuilt Linux binaries:

- `ghcr.io/iamwavecut/ninfer-all:cuda12.9`: CUDA 12.9.2 with `70;75;80;86;89`, including Volta.
- `ghcr.io/iamwavecut/ninfer-all:cuda13.4`: CUDA 13.4.3 with `75;80;86;89;120a`, including Blackwell.

Each image carries all listed architectures in one executable. Both support mixed-GPU pipelines
within their architecture set. Choose the variant for your GPUs and installed NVIDIA driver.
The host also needs NVIDIA Container Toolkit. For a CUDA 13 host, first check GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:13.4.3-base-ubuntu26.04 nvidia-smi
```

Download and serve the Qwen3.8 model with the measured `tuned` profile:

```bash
docker run --rm -v "$PWD/models:/models" ghcr.io/iamwavecut/ninfer-all:cuda13.4 download qwen38-27b
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" ghcr.io/iamwavecut/ninfer-all:cuda13.4 run qwen38-27b
```

The API is available at `http://127.0.0.1:8080/v1`. `docker build --build-arg ARCHS=86 -t ninfer .`
builds a separate Ubuntu image from source with the `sm_86` build alone; the [README](../README.md#docker)
covers the container's commands, volumes and compose file, and the [launch recipes](recipes.md)
the command for each published artifact.

## Native Ubuntu 24.04 build

Install the CUDA Toolkit from NVIDIA (12.8 or newer, 13.1 or newer for `120a`).
Then install the host compiler and media dependencies:

```bash
sudo apt-get update
sudo apt-get install --yes \
  build-essential gcc-13 g++-13 cmake ninja-build pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev
```

Select GCC 13 for host and CUDA compilation:

```bash
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDAHOSTCXX=/usr/bin/g++-13
```

Configure and build the Linux applications:

```bash
cmake -S . -B build-sm86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DNINFER_BUILD_APPS=ON \
  -DBUILD_TESTING=OFF \
  -DNINFER_BUILD_BENCHMARKS=OFF

cmake --build build-sm86 --parallel 2
```

For another GPU family, set `CMAKE_CUDA_ARCHITECTURES` from the table above in a build directory
of its own. The build creates these applications:

```text
build-sm86/apps/ninfer
build-sm86/apps/ninfer-serve
build-sm86/apps/ninfer-calibrate
build-sm86/apps/ninfer-perplexity
```

## Turing (sm_75)

A `75` build runs on the TITAN RTX and the RTX 20 series. The container image carries one; a native
build needs CUDA 12.8 or newer and `-DCMAKE_CUDA_ARCHITECTURES=75`:

```bash
cmake -S . -B build-sm75 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DNINFER_BUILD_APPS=ON

cmake --build build-sm75 --parallel
```

Turing's tensor cores multiply FP16 and INT8 but not BF16 or TF32, it has no `cp.async`, and a
block has 64 KiB of shared memory. A `75` build runs the kernels of the other targets with Turing
forms of those instructions and Turing shapes of the kernels that do not fit
([Turing tier](maintainer/build-system.md#turing-tier-75)):

- Integer routes (the T2 ternary projections, A8 activations, INT8-family attention, GGUF `dp4a`
  and MMQ kernels) and FP16 routes run on the native tensor cores.
- BF16 products run on FP16 tensor cores for every output whose operands are FP16 values, and in
  FP32 for the others; no value is narrowed. BF16-cache attention multiplies Q, K and V as their
  FP16 roundings, and every attention PV product with BF16 values rounds the probabilities to
  FP16.
- Every model, KV format and drafter runs. The Linear routes come from the legacy table only, and
  `NINFER_LINEAR_ROUTES=unified` is refused.

Device route profiles of the TITAN RTX and the other Turing cards the
[card table](gpus.md#turing-75-build-75) marks are built in; another Turing part calibrates at first
start (see [device profiles](device-profiles.md)). What runs and how fast is in the
[TITAN RTX measurements](performance/titan-rtx-2026-10.md).

## Cards of several architectures in one pipeline

`CMAKE_CUDA_ARCHITECTURES` may list several architectures. The binary then carries device code for
each, so a layer pipeline (`--devices`, `--stage-layers`) can span cards of different
architectures: for example a Tesla V100, a TITAN RTX and RTX 30 or 40 series cards with CUDA 12.8,
or a TITAN RTX, RTX 30 and 40 series cards and an RTX 50 series card with CUDA 13.1:

```bash
cmake -S . -B build-cu128-multi -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  "-DCMAKE_CUDA_ARCHITECTURES=70;75;80;86;89" \
  -DNINFER_BUILD_APPS=ON

cmake --build build-cu128-multi --parallel
```

- `70` needs CUDA 12.x and `120a` needs CUDA 13.1 or newer, so no build holds both.
- Each card runs the code of the newest listed architecture it can run, with the kernels, routes,
  route catalog and built-in device profile of its own single-architecture build
  ([code tiers](maintainer/build-system.md#code-tiers)). On one card the binary gives the same
  tokens as that card's own build.
- Each card computes its own layers with its own arithmetic (for example, Turing attention rounds
  BF16 values and probabilities to FP16), so greedy output across a mixed pipeline may diverge from
  any one card's after some tokens.
- Compile time and binary size grow with the number of architectures.

A build of one architecture also pipelines cards of different compute capabilities when its code
runs on each of them, for example an `86` build across RTX 30 and RTX 40 series cards.

## Tesla V100 (sm_70)

A `70` build runs on Tesla V100 (16 or 32 GB). It needs CUDA 12.8 or 12.9: CUDA 13 no longer
compiles `sm_70`, and configure refuses that combination. Configure it like any other build, with
`-DCMAKE_CUDA_ARCHITECTURES=70` and the 12.x `nvcc`:

```bash
cmake -S . -B build-sm70 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DNINFER_BUILD_APPS=ON

cmake --build build-sm70 --parallel 2
```

Volta's tensor cores multiply FP16 only (`mma.m8n8k4`): there are no BF16 or integer tensor cores,
no `cp.async` and no `ldmatrix`. A `70` build therefore runs its own routes where they decide the
speed, and the kernels written for Ampere everywhere else, on emulated warp primitives
(`src/ops/common/sm70_emulation.cuh`), which are correct but slower:

- **Dense projections** of every native format (T2, Q2, Q4, Q5, Q6, Q8, BF16, FP8, NVFP4) up to 64
  columns: a lane decodes the weights of its row straight into FP16 tensor-core fragments, and a
  row-split group's FP32 products are scaled when the group ends. Up to eight columns read the BF16
  activations directly; wider calls convert them to FP16 once. From 65 columns
  (`NINFER_SM70_CUBLAS_MIN`) the weights are dequantized to FP16 32 MiB at a time and multiplied by
  cuBLAS with FP32 accumulation; a call of more than 8,192 columns (a Vision item's patches) passes
  over them 8,192 at a time.
- **T2 projections at one to eight columns** (decode, two requests, an MTP or DFlash2 verify round)
  take an integer route where the projection's policy admits integer activations, as the other
  targets' T2 A8 route does: the activations are quantized to int8 with one scale per 64 values,
  and `dp4a` multiplies them by the ternary codes, 10-35% faster than the tensor-core route. The
  MLP `gate_up` takes it only under `--mlp-a8-decode`, the flag that asks for an integer gate_up at
  these widths; on Volta that applies to a T2 `gate_up`, and a Q4 one, which has no integer route
  here, stays A16.
- **Weight layout**: the row-split projection weights of the text and MTP layers (T2, Q2, Q4, Q5,
  Q6) load in the panel layout, four rows' records of a group per 128-byte line, which the
  decode reads stream at a larger share of the card's bandwidth.
- **GGUF projections** keep NInfer's DP4A vector kernel up to eight columns and ggml's DP4A matrix
  kernel above (as llama.cpp does on Volta); GGUF dense projections from 65 columns take the cuBLAS
  route.
- **Attention** of every KV format runs on Volta's tensor cores with FP32 accumulation. Decode and
  verification use a Volta kernel that decodes each format's keys and values to FP16 as it loads
  them (32 dimensions of a quantized row a load), with the same fused append, splits and reducers
  as the other targets. Prompt attention gives a CTA 64 query rows of one KV head and decodes each
  32-key tile of the visible pages into shared memory once for all of them; it reads the pages
  where they are and needs no workspace.
- **BF16 values beyond FP16's range.** Volta's tensor cores take FP16 operands, which saturate at
  +-65504. Where a BF16 activation of a dense projection, or a query, key or value of attention
  over a BF16 cache, is beyond that (magnitude 2^16 or more), the conversion flags its 64-value
  chunk, column or key tile, and the product adds in FP32 what the saturation took: a projection
  adds the weights times the excess, and attention forms the tile's scores again from the BF16
  values and adds the probabilities times the values' excess. Data within FP16's range takes the
  tensor-core paths unchanged. Values below FP16's normal range keep FP16's absolute precision
  (2^-25), and the queries of a quantized cache, which are rotated in FP16, still saturate.
- FP8 A8 and NVFP4 W4A4 activations are not available, as on `sm_86`; those weights run their A16
  routes.

On a 16 GB card the official Qwen3.8-27B artifact loads with `--embedding-q4 --lm-head-q4` (its
Q8 embedding and output head stored as Q4 while loading, 14.7 GiB of weights). What else fits is in
the [measurements](performance/tesla-v100-2026-10.md#models-on-a-16-gb-card).

The V100-SXM2-16GB and V100-SXM2-32GB device route profiles are built in; another V100 part
calibrates at first start, in about nine seconds (see [device profiles](device-profiles.md)).

## Optional vcpkg dependencies

Linux can use the pinned `vcpkg.json` manifest instead of system FFmpeg and curl packages.
Bootstrap the pinned vcpkg revision:

```bash
git clone https://github.com/microsoft/vcpkg.git "$HOME/.local/share/vcpkg"
git -C "$HOME/.local/share/vcpkg" checkout 4bca8fd8654e5ba76f92661db7bfe954768ad8ef
"$HOME/.local/share/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
```

Add these options to the native CMake command:

```text
-DCMAKE_TOOLCHAIN_FILE=$HOME/.local/share/vcpkg/scripts/buildsystems/vcpkg.cmake
-DVCPKG_TARGET_TRIPLET=x64-linux
```

## Build options

Every option below is off by default; a default build runs the tested routes. Pass them to the
CMake configure command as `-DNAME=VALUE`.

| option | effect |
|---|---|
| `NINFER_SFU_SIGMOID_SILU=ON` | sigmoid and SiLU evaluate `ex2.approx` and a correctly rounded reciprocal on the SFU instead of `expf` and a divide |
| `NINFER_SFU_SOFTPLUS=ON` | the GDN decay gate's softplus runs on the SFU as well, switching to a log1p series where e^x is below 1/16 |
| `NINFER_BF16_RESIDUAL_ADD=ON` | BF16 linear projections are added onto the residual in bf16 instead of fp32 |
| `NINFER_NVCC_SPLIT_COMPILE=N` | nvcc optimizes one translation unit on N threads (`--split-compile`); `0` is every core, which a parallel ninja multiplies by its job count |
| `NINFER_PTXAS_VERBOSE=ON` | ptxas reports each kernel's registers and local-memory spills |
| `NINFER_WEBUI_DIR=PATH` | compiles the WebUI in that directory (an `index.html` or `index.html.gz` at its root) into the server |
| `NINFER_SM120_NATIVE=ON` | on a `120a` build, compiles upstream's native routes instead of the `mma.sync` path (every `120a` build compiles the FP8 A8 and NVFP4 W4A4 units) |
| `NINFER_PDL=ON` | on a `120a` build of the `mma.sync` path, launches decode-graph kernels as programmatic dependents, so a kernel stages its weights while the one before it finishes (the native routes always do) |
| `NINFER_TMA_STAGING=ON` | passes the NVFP4 TMA descriptors through device memory with a capturable staging kernel, as Windows builds must, so that path can be tested on Linux (`120a`) |
| `NINFER_MULTICALL=ON` | Linux: links `ninfer`, `ninfer-serve`, `ninfer-calibrate` and `ninfer-perplexity` into one executable, `apps/ninfer-multicall`, with the four names as symlinks to it, so the kernel image (about 0.5 GB per architecture) is there once instead of four times; the container image is built this way |
| `NINFER_MEDIA_NATIVE_PNG=ON` | decodes PNG images natively instead of through FFmpeg, as Windows builds do by default, so that decoder can be tested on Linux |
| `NINFER_DIRECTSTORAGE=ON` | Windows: fetches the DirectStorage 1.3 runtime and offers `--disk-kv-directstorage` for disk-tier restores |

The Windows option compiles against Windows headers but has not been run on this line.
At run time, `NINFER_CUDA_SYNC=spin|blocking|yield|auto` selects the CUDA synchronization schedule
(unset keeps CUDA's default) and `NINFER_T2_A8_TILE=off` returns the ternary prefill route to its
64-row kernel.

## Bash scripts

`scripts/` holds the download script, the launcher and the packager, each with a Windows twin.
The download script saves models under `scripts/models` by default and checks their size and
SHA-256:

```bash
./scripts/download-model.sh qwen38-27b
./scripts/download-model.sh qwen36-35b-a3b
./scripts/download-model.sh qwen36-27b
```

Set `NINFER_MODEL_DIR` to use another model directory. `run.sh` serves either model, and
`NINFER_MODEL` points it at a model path anywhere:

```bash
./scripts/run.sh qwen38-27b
./scripts/run.sh qwen36-35b-a3b
NINFER_MODEL=/path/to/qwen3_8_27b.ninfer ./scripts/run.sh qwen38-27b int8
```

The launcher uses `build-linux/apps/ninfer-serve` when the executable is not beside the script.
Set `NINFER_SERVER` to select another executable. Profiles and the measurements behind them are in
[launcher profiles](maintainer/launcher-profiles.md).

`scripts/package-release.sh` creates split Linux `tar.gz` volumes; `scripts/package-release.ps1`
creates Windows 7-Zip volumes. Names contain the platform, CUDA minor version, architecture list
and source commit, without the product `VERSION`. `build.sh --package` and `build.ps1 -Package`
run the packagers after a build. The Native release workflow builds Debian 13 and Windows packages
with CUDA 12.9.2 and 13.4.3. See [cutting a release](maintainer/release-process.md).

## Validation

Make sure that the applications start:

```bash
./build-sm86/apps/ninfer --help
./build-sm86/apps/ninfer-serve --help
```

Run one short generation with the real Qwen3.8 artifact:

```bash
./build-sm86/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain prefill and decode in two sentences." \
  --max-context 8192 --max-new 32 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

A successful compile does not qualify Linux performance.
Record the GPU, driver, CUDA Toolkit, compiler, workload, and result before you publish Linux measurements.
