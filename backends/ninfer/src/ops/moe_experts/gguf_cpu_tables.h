#pragma once

// ggml's i-quant grids, sign patterns and IQ4_NL values (third_party/ggml-quants, MIT) as host
// constants, for the CPU expert products. Each including translation unit gets its own copy.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
