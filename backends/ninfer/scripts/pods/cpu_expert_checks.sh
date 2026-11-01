#!/usr/bin/env bash
set -Eeuo pipefail
root=${NINFER_SOURCE_ROOT:-/workspace/ninfer-work/src}
cxx=${CXX:-g++}
cd "$NINFER_JOB_DIR"
uname -srm
"$cxx" --version | head -n 1
flags=(-std=c++20 -O3 -Wall -Wextra -Werror -I "$root/include" -I "$root/src")
if [ "$(uname -s)" = Darwin ]; then flags+=(-arch "$(uname -m)"); fi
objects=()
dispatch=()
if [ "$(uname -m)" = x86_64 ]; then
    dispatch=(-DNINFER_CPU_EXPERT_X86=1)
    "$cxx" "${flags[@]}" -mavx2 -c "$root/src/ops/moe_experts/moe_expert_cpu_avx2.cpp" -o avx2.o
    "$cxx" "${flags[@]}" -mavx512f -mavx512bw -mavx512vl -mavx512vnni \
        -c "$root/src/ops/moe_experts/moe_expert_cpu_avx512.cpp" -o avx512.o
    objects=(avx2.o avx512.o)
fi
"$cxx" "${flags[@]}" "${dispatch[@]}" "$root/src/ops/moe_experts/moe_expert_cpu.cpp" \
    "$root/tests/ops/test_moe_expert_cpu.cpp" "${objects[@]}" -o cpu-expert-test
./cpu-expert-test
