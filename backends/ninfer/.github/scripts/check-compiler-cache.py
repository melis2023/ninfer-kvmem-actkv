#!/usr/bin/env python3
"""Check object-cache reuse and invalidation of CUDA host-only source changes."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import uuid

from build_telemetry import cache_stats


def run(*args):
    subprocess.run(args, check=True)


root = Path("build-cache-check")
root.mkdir(exist_ok=True)
(root / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.28)
project(cache_check LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
add_library(cache_check OBJECT host.cpp kernel.cu)
target_include_directories(cache_check PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(cache_check PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xfatbin=-compress-all;-compress-mode=size>")
""")
# Restored shards can contain every probe variant from this same workflow run.
# Give each invocation fresh source content so invalidation must cause a miss.
identity = uuid.uuid4().hex
(root / "host.cpp").write_text(f'#include <cuda_runtime.h>\nint host_{identity}() {{ return 42; }}\n')
(root / "host_only.h").write_text("#define HOST_VALUE 20\n")
(root / "kernel.cu").write_text(
    f'__global__ void kernel_{identity}(int *p) {{ p[threadIdx.x] = 42; }}\n'
    '#ifndef __CUDA_ARCH__\n#include "host_only.h"\nint host_only() { return 17 + HOST_VALUE; }\n#endif\n')
run("cmake", "-S", str(root), "-B", str(root / "out"), "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=" + os.environ["ARCHS"],
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache", "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
run("ccache", "--zero-stats")
run("cmake", "--build", str(root / "out"), "-j")
run("ccache", "--zero-stats")
# New checkout timestamps must not invalidate unchanged content.
for source in (root / "host.cpp", root / "kernel.cu"):
    os.utime(source, None)
run("cmake", "--build", str(root / "out"), "--clean-first", "-j")
stats = cache_stats()
(root / "cache-stats.json").write_text(json.dumps(stats, indent=2) + "\n")
hits = stats["direct_cache_hit"] + stats["preprocessed_cache_hit"]
print(f"Unchanged C++/CUDA: {hits} cache hits, {stats['cache_miss']} misses", flush=True)
if hits != 2 or stats["cache_miss"]:
    sys.exit("C++/CUDA object-cache reuse failed; do not start the long build")
# nvcc preprocessing defines __CUDA_ARCH__; changes outside that branch must also invalidate.
run("ccache", "--zero-stats")
kernel = root / "kernel.cu"
kernel.write_text(kernel.read_text().replace("return 17", "return 18"))
run("cmake", "--build", str(root / "out"), "-j")
if cache_stats()["cache_miss"] != 1:
    sys.exit("CUDA host-only source change did not invalidate its cached object")
run("ccache", "--zero-stats")
run("ccache", "--zero-stats")
(root / "host_only.h").write_text("#define HOST_VALUE 21\n")
run("cmake", "--build", str(root / "out"), "--clean-first", "-j")
if cache_stats()["cache_miss"] != 1:
    sys.exit("CUDA host-only header change did not invalidate its cached object")
if os.name != "nt":
    # Change a private copy at the same path: changing only its path would not
    # establish that ccache hashes the host compiler's content.
    host_tools = (root / "host-tools").resolve()
    host_tools.mkdir(exist_ok=True)
    # GCC resolves its internal programs and standard headers relative to bin.
    # Retain that layout while keeping private, independently mutable drivers.
    for name in ("include", "lib", "libexec"):
        sibling = root / name
        if not sibling.exists():
            sibling.symlink_to(Path("/usr") / name, target_is_directory=True)
    for name in ("gcc", "g++"):
        shutil.copy2(shutil.which(name), host_tools / name)
    source = root / "host_compiler.cu"
    source.write_text(f'__global__ void host_compiler_{identity}(int *p) {{ *p = 7; }}\n')
    command = ("ccache", "nvcc", "-ccbin", str(host_tools), "-MD", "-c", str(source),
               "-o", str(root / "host_compiler.o"))
    run(*command)
    run("ccache", "--zero-stats")
    run(*command)
    if cache_stats()["direct_cache_hit"] != 1:
        sys.exit("Unchanged CUDA host compiler did not reuse its cached object")
    for name in ("gcc", "g++"):
        # ELF ignores bytes after its segments; the compiler remains executable.
        with (host_tools / name).open("ab") as output:
            output.write(b"\nccache host compiler content probe\n")
    run("ccache", "--zero-stats")
    run(*command)
    if cache_stats()["cache_miss"] != 1:
        sys.exit("CUDA host compiler content change did not invalidate its cached object")
run("ccache", "--zero-stats")
