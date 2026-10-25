# syntax=docker/dockerfile:1.7
#
# NInfer server image. ninfer, ninfer-serve, ninfer-perplexity and ninfer-calibrate are built once
# per architecture -- sm_86 (Ampere; the same cubins run on Ada sm_89) and sm_120a (consumer
# Blackwell), since a NInfer build targets exactly one -- as one multi-call executable each
# (NINFER_MULTICALL), next to the launcher and the downloader. The entrypoint runs the build that
# matches the GPU it is given (docker/entrypoint.sh).
#
#   docker build -t ninfer .                               both architectures, from source
#   docker build --build-arg ARCHS=86 -t ninfer .          one architecture
#   docker build --build-context dist=<dir> -t ninfer .    package prebuilt binaries (what CI does)
#
# A prebuilt dist directory has the layout docker/stage-dist.sh writes: bin/sm<arch>/ and
# packages/sm<arch>.txt for each architecture.

ARG CUDA_VERSION=13.4.2
ARG UBUNTU_VERSION=26.04

# --- toolchain: also the CI build environment ------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS toolchain
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ccache \
        cmake \
        file \
        git \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
        python3 \
        python3-jinja2 \
        rsync \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# --- build -----------------------------------------------------------------------------------------
FROM toolchain AS build
ARG ARCHS="86 120a"
ARG JOBS=""
WORKDIR /src
COPY . .

# Keep source mtimes stable only when contents match; restored/older checkouts must still rebuild
# changed inputs. Packages, the Dockerfile, the build script, CMake scripts and the architecture
# identify the configuration: removed flags and changed defaults must not reuse CMakeCache.txt.
# Lock the mutable Ninja trees and copy deliverables out of the transient cache mount.
RUN --mount=type=cache,id=ninfer-build,target=/build,sharing=locked \
    --mount=type=cache,target=/ccache \
    export CCACHE_DIR=/ccache CCACHE_MAXSIZE=20G NINFER_TESTS=OFF NINFER_MULTICALL=ON \
    && if [ -n "$JOBS" ]; then export NINFER_JOBS="$JOBS"; fi \
    && find . -type f \( -path ./Dockerfile -o -path ./scripts/build-native.sh \
        -o -name CMakeLists.txt -o -name '*.cmake' \) -exec sha256sum {} + > /build/configuration \
    && dpkg-query -W >> /build/configuration \
    && LC_ALL=C sort -o /build/configuration /build/configuration \
    && rsync --recursive --links --checksum --delete /src/ /build/src/ \
    && for arch in $ARCHS; do \
         build_dir="/build/$( { cat /build/configuration; echo "arch $arch"; } | sha256sum | cut -d ' ' -f 1)" \
         && NINFER_CUDA_ARCH="$arch" NINFER_BUILD_DIR="$build_dir" /build/src/scripts/build-native.sh configure \
         && NINFER_BUILD_DIR="$build_dir" /build/src/scripts/build-native.sh target ninfer-multicall \
         && /build/src/docker/stage-dist.sh "$build_dir" "$arch" /dist || exit 1; \
       done \
    && ccache --show-stats

FROM scratch AS dist
COPY --from=build /dist/ /

# --- runtime ---------------------------------------------------------------------------------------
# Plain Ubuntu, not a CUDA image: the binaries need only cuBLAS and the CUDA runtime from the
# toolkit, which their package lists name and NVIDIA's apt repository provides, and the driver
# library comes from the host through the NVIDIA Container Toolkit. A CUDA base image would add
# every other CUDA library (about 1.4 GB compressed) and forward-compatibility libraries that break
# GeForce cards.
FROM ubuntu:${UBUNTU_VERSION} AS runtime
ARG UBUNTU_VERSION
ARG DEBIAN_FRONTEND=noninteractive
# The libraries first, from the package lists alone: this layer changes with a dependency, not with
# every build of the binaries, so an update pulls little more than the binaries.
COPY --from=dist /packages/ /tmp/ninfer-packages/
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl \
    && curl -fsSLo /tmp/cuda-keyring.deb \
       "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(echo "$UBUNTU_VERSION" | tr -d .)/x86_64/cuda-keyring_1.1-1_all.deb" \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update \
    && sort -u /tmp/ninfer-packages/*.txt | xargs apt-get install --yes --no-install-recommends aria2 \
    && find /usr/local -name 'libcublas.so.*' -printf '%h\n' | sort -u > /etc/ld.so.conf.d/ninfer-cuda.conf \
    && ldconfig \
    && apt-get purge --yes cuda-keyring \
    && rm -rf /var/lib/apt/lists/* /tmp/ninfer-packages /tmp/cuda-keyring.deb

# --chmod: a dist that went through a CI artifact has lost its executable bits, and a chmod in a
# later layer would store every binary a second time.
COPY --from=dist --chmod=0755 /bin/ /opt/ninfer/
RUN for arch in /opt/ninfer/sm*; do \
      for name in ninfer ninfer-serve ninfer-calibrate ninfer-perplexity; do \
        ln -s ninfer-multicall "$arch/$name"; \
      done; \
    done
COPY scripts/run.sh scripts/download-model.sh /opt/ninfer/
COPY docker/entrypoint.sh /usr/local/bin/ninfer-entrypoint

# nvidia-smi (the "utility" capability) is how the entrypoint reads the GPU's compute capability.
# Any driver of the CUDA 13 branch runs these binaries through minor-version compatibility.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0" \
    NINFER_MODEL_DIR=/models \
    NINFER_DEVICE_PROFILES=/cache/device-profiles.json \
    NINFER_HOST=0.0.0.0 \
    NINFER_PORT=8080
VOLUME ["/models", "/grafts", "/cache"]
WORKDIR /models
EXPOSE 8080
STOPSIGNAL SIGTERM
ENTRYPOINT ["ninfer-entrypoint"]
CMD ["run", "qwen38-27b"]
