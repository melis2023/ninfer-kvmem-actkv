#pragma once

// One answer, for every test that needs a CUDA device and skips (exit 77) without one, to whether
// a cudaGetDeviceCount failure means this machine has no usable device: no GPU, a driver older than
// the runtime, or only the toolkit's stub libcuda -- which is what a container or CI runner without
// a GPU loads. Any other failure is a real error.

#include <cuda_runtime.h>

namespace ninfer::test {

[[nodiscard]] inline bool cuda_unavailable(cudaError_t error) noexcept {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
           error == cudaErrorStubLibrary;
}

} // namespace ninfer::test
