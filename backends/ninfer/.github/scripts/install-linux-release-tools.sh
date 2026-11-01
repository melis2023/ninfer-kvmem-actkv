#!/usr/bin/env bash
# Build on Debian 13; install the Toolkit without a GPU driver.
set -euo pipefail
: "${CUDA_VERSION:?}"
minor="${CUDA_VERSION%.*}"
suffix="${minor/./-}"
apt-get update
apt-get install -y --no-install-recommends \
  build-essential ninja-build pkg-config patchelf python3 python3-venv \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev \
  libcurl4-openssl-dev zlib1g-dev
if [[ "$CUDA_VERSION" == 12.9.2 ]]; then
  # Debian 13 rejects the SHA-1 binding in NVIDIA's Debian 12 repository key.
  # Use NVIDIA's release installer instead; do not weaken APT signature policy.
  installer=/tmp/cuda-toolkit.run
  curl --retry 3 -fL -o "$installer" \
    https://developer.download.nvidia.com/compute/cuda/12.9.2/local_installers/cuda_12.9.2_575.57.08_linux.run
  sh "$installer" --silent --toolkit --toolkitpath="/usr/local/cuda-$minor"
  rm -f "$installer"
else
  curl -fsSLo /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i /tmp/cuda-keyring.deb
  apt-get update
  apt-get install -y --no-install-recommends "cuda-toolkit-$suffix=$CUDA_VERSION-1"
fi
python3 -m venv /opt/ninfer-bootstrap
/opt/ninfer-bootstrap/bin/pip install --disable-pip-version-check uv==0.11.11
/opt/ninfer-bootstrap/bin/uv python install 3.11
/opt/ninfer-bootstrap/bin/uv venv --clear --python 3.11 --seed /opt/ninfer-build-tools
/opt/ninfer-build-tools/bin/pip install --disable-pip-version-check cmake==3.31.10
if [[ "$CUDA_VERSION" == 12.9.2 ]]; then
  /opt/ninfer-build-tools/bin/python3.11 .github/scripts/patch-cuda12-glibc.py "/usr/local/cuda-$minor"
fi
echo '/opt/ninfer-build-tools/bin' >> "$GITHUB_PATH"
echo "/usr/local/cuda-$minor/bin" >> "$GITHUB_PATH"
echo "CUDACXX=/usr/local/cuda-$minor/bin/nvcc" >> "$GITHUB_ENV"
echo "CUDA_PATH=/usr/local/cuda-$minor" >> "$GITHUB_ENV"
