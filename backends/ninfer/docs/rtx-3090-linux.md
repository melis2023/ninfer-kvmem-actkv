# Build NInfer on Linux

This guide runs NInfer on Linux from the published container image, or builds it natively.
`CMAKE_CUDA_ARCHITECTURES` selects one target per build directory:

| value | GPUs | toolkit |
|---|---|---|
| `86` (the default) | RTX 30 series; an `86` build also runs on the RTX 40 series | CUDA 12.8 or newer |
| `89` | RTX 40 series only | CUDA 12.8 or newer |
| `120a` | RTX 50 series and RTX PRO 6000 Blackwell | CUDA 13.1 or newer (configure refuses 12.8 and 12.9, which miscompile `sm_120a` kernels) |
| `80` | A100 class (GA100), an unmeasured compatibility target | CUDA 12.8 or newer |

An `89` or `120a` build does not run on an RTX 3090.

## Container image

The published image, `ghcr.io/iamwavecut/ninfer-all:latest`, is the shortest path on Bazzite and
other Linux distributions: CUDA 13.4 on Ubuntu 26.04, with an `sm_86` build for the RTX 30 and 40
series and an `sm_120a` build for Blackwell, and it starts the one that matches the GPU. It needs a
driver of the CUDA 13 branch (580 or newer) and the NVIDIA Container Toolkit; make sure that Docker
can access the GPU first:

```bash
docker run --rm --gpus all nvidia/cuda:13.4.2-base-ubuntu26.04 nvidia-smi
```

Download and serve the Qwen3.8 model with the measured `tuned` profile:

```bash
docker run --rm -v "$PWD/models:/models" ghcr.io/iamwavecut/ninfer-all download qwen38-27b
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" ghcr.io/iamwavecut/ninfer-all run qwen38-27b
```

The API is available at `http://127.0.0.1:8080/v1`. `docker build --build-arg ARCHS=86 -t ninfer .`
builds the same image from source with the `sm_86` build alone; the [README](../README.md#docker)
covers the container's commands, volumes and compose file, and its [Running](../README.md#running)
section the command for each published artifact.

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
| `NINFER_D3D12_RESIDENCY=ON` | Windows: offers `--wddm-evictable-budget`, device arenas from a D3D12 heap held resident |
| `NINFER_DIRECTSTORAGE=ON` | Windows: fetches the DirectStorage 1.3 runtime and offers `--disk-kv-directstorage` for disk-tier restores |

The Windows options compile against Windows headers but have not been run on this line.
At run time, `NINFER_CUDA_SYNC=spin|blocking|yield|auto` selects the CUDA synchronization schedule
(unset keeps CUDA's default) and `NINFER_T2_A8_TILE=off` returns the ternary prefill route to its
64-row kernel.

## Bash scripts

The `scripts/` directory contains Bash versions of each Windows download, launcher, and packaging
script. Download scripts save models under `scripts/models` by default:

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

`scripts/package-release.sh` creates the Linux `tar.gz` archive for the release named in `VERSION`,
and `scripts/package-release.ps1` the Windows ZIP; `build.sh --package` and `build.ps1 -Package` run
them after a build.

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
