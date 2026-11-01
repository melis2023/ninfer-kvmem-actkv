# NInfer Linux archive

This archive contains prebuilt x64 binaries. Single-CUDA names record the Toolkit, GPU architectures and source commit.
Universal names use `cuda12.9-13.4-<commit12>` and contain both backends. `BUILD-INFO.json` records the complete revision, compiler, build
system and runtime requirements. `VERSION` is source metadata, not the archive identifier.

## Extract and verify

Download every numbered volume and the matching `SHA256SUMS-*.txt` for one variant. For example,
replace `<commit12>` below with the commit in the downloaded filenames:

```bash
name='ninfer-linux-x64-cuda12.9-13.4-<commit12>'
sha256sum -c "SHA256SUMS-$name.txt"
cat "$name".tar.gz.[0-9][0-9][0-9] | tar -xz
cd "$name"
sha256sum -c SHA256SUMS.txt
```

CUDA 12.9 packages contain `sm_70`, `sm_75`, `sm_80`, `sm_86` and `sm_89`.
CUDA 13.4 packages contain `sm_75`, `sm_80`, `sm_86`, `sm_89` and `sm_120a`.
The universal package selects CUDA 12 for Volta and CUDA 13 for consumer Blackwell. On overlapping
targets it prefers CUDA 13 if the driver supports it, otherwise CUDA 12. Use `--cuda-backend 12`
or `--cuda-backend 13` to require one backend. Every GPU in `--devices` must fit that backend.
V100 plus Blackwell cannot share a process with these sets. Install a compatible NVIDIA driver
on the host. The minimum Driver API family is 12.0 for CUDA 12 and 13.0 for CUDA 13. This is a
compatibility admission check, not GPU qualification of every driver release.
The CUDA 12 build applies a glibc 2.41 exception-declaration patch to NVIDIA headers;
`BUILD-INFO.json` records it. Function bodies are not modified.

## Debian runtime

CI builds on Debian 13 (glibc 2.41). CUDA runtime libraries are included under `backends/cuda12/` or `backends/cuda13/`, with NVIDIA's EULA. System libraries remain dynamically linked. Newer distributions must also provide the
recorded FFmpeg and other library SONAMEs; newer glibc alone is insufficient. Local builds can use
another Debian-family host; check `BUILD-INFO.json`.

On Debian 13, install the recorded system runtime packages:

```bash
sudo apt-get update
xargs -a packages/multi.txt sudo apt-get install -y ca-certificates curl
./ninfer --help
./ninfer-serve --help
./ninfer_bench --help
```

This installs FFmpeg, libcurl and the other resolved system dependencies. It does not install a
CUDA Toolkit or a driver. The build's temporary driver stub is never shipped. The executables load
one private runtime from the selected backend directory. All commands share that backend's kernel
images. Keep `backends/` and `help/` beside the executables when moving the installation.
`--help` and `--version` work without a driver. NVIDIA libraries remain subject to the
`NVIDIA-CUDA-EULA.txt` in their backend directory, separately from NInfer's Apache-2.0 licence.

## Docker without compilation

From an extracted **single-CUDA** Debian 13 archive (the universal archive has no Dockerfile):

```bash
docker build -t ninfer:local .
docker run --rm --gpus all -p 8080:8080 -v ninfer-models:/models ninfer:local
```

The included Dockerfile installs runtime packages and copies these binaries. It does not build
NInfer again. The host needs NVIDIA Container Toolkit for GPU access. This image uses the same
entrypoint commands, `run.sh` launcher and model profiles as the source tree. The Container image
workflow packages both Native release archives as `cuda12.9` and `cuda13.4`, with matching
`cuda12.9-<commit12>` and `cuda13.4-<commit12>` tags, without recompilation.

## Start a model

```bash
./run.sh qwen38-27b
```

The launcher downloads the selected model on first use. Set `NINFER_MODEL_DIR` to choose storage,
`NINFER_MODEL` to use an existing artifact, and `NINFER_HOST`/`NINFER_PORT` for the server address.
CI checks device images, dynamic dependencies and `--help`; it does not run GPU inference or
measure performance. Verify the intended model and GPU before relying on a package for serving.
