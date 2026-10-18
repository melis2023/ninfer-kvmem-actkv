#pragma once
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>

namespace kvmem {
inline double parse_lrfu_real(const char* text, bool allow_zero) {
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end || !std::isfinite(value) || value > 1 ||
        (allow_zero ? value < 0 : value <= 0))
        throw std::invalid_argument("LRFU value must be " + std::string(allow_zero ? "in [0,1]" : "in (0,1]"));
    return value;
}
inline int parse_score_aggregation(const char* text) {
    const std::string value(text);
    if (value == "mean") return 0;
    if (value == "max") return 1;
    throw std::invalid_argument("KVMem score aggregation must be mean or max");
}
} // namespace kvmem
