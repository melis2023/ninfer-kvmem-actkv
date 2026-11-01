#!/usr/bin/env bash
set -Eeuo pipefail
# Add the build tools to a retained runtime pod without replacing its Python environment.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq cmake ninja-build build-essential pkg-config \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev \
    libswresample-dev libcurl4-openssl-dev
cmake --version
nvcc --version
df -h /workspace /dev/shm
