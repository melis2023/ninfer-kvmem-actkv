#pragma once
#include <cstdlib>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

namespace kvmem {
enum class QueryScorer : uint8_t { PageSoftmax = 0, MeanQCosine = 1 };
inline const char* query_scorer_name(QueryScorer scorer) {
    return scorer == QueryScorer::MeanQCosine ? "meanq-cosine" : "page-softmax";
}
inline QueryScorer parse_query_scorer(const char* text) {
    const std::string value(text);
    if (value == "page-softmax") return QueryScorer::PageSoftmax;
    if (value == "meanq-cosine") return QueryScorer::MeanQCosine;
    throw std::invalid_argument("KVMem Query scorer must be meanq-cosine or page-softmax");
}
inline double parse_graph_alpha(const char* text) {
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end || std::isspace(static_cast<unsigned char>(*text)) ||
        !std::isfinite(value) || value < 0 || value > 1)
        throw std::invalid_argument("KVMem graph alpha must be finite and in [0,1]");
    return value;
}
inline void validate_query_scoring(QueryScorer scorer, int aggregation, double alpha) {
    if ((scorer != QueryScorer::PageSoftmax && scorer != QueryScorer::MeanQCosine) ||
        (aggregation != 0 && aggregation != 1) || !std::isfinite(alpha) || alpha < 0 || alpha > 1)
        throw std::invalid_argument("invalid KVMem Query scoring configuration");
}
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
