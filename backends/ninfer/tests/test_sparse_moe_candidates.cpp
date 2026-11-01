#include "ops/sparse_moe/candidate.h"
#include "ops/sparse_moe/policy.h"

#include <iostream>
#include <stdexcept>

int main() {
    using namespace ninfer::ops::detail;
    int failures = 0;
    const auto expect = [&](bool value, const char* message) {
        if (!value) { std::cerr << message << '\n'; ++failures; }
    };
    for (const char* name : {"auto", "decode", "small-t", "prefill", "prefill-narrow", "prefill-wide"}) {
        const auto candidate = parse_sparse_moe_candidate(name);
        expect(sparse_moe_candidate_name(candidate) == name, "candidate name does not round trip");
        expect(!sparse_moe_candidate_supports_tokens(candidate, 0), "zero tokens admitted");
        expect(!sparse_moe_candidate_supports_tokens(candidate, -1), "negative tokens admitted");
    }
    expect(!sparse_moe_candidate_supports_tokens(SparseMoeCandidate::SmallT, 1), "small-T admitted T=1");
    expect(sparse_moe_candidate_supports_tokens(SparseMoeCandidate::SmallT, 2), "small-T rejected T=2");
    expect(sparse_moe_candidate_supports_tokens(SparseMoeCandidate::SmallT, 46), "small-T rejected T=46");
    expect(!sparse_moe_candidate_supports_tokens(SparseMoeCandidate::SmallT, 47), "small-T admitted T=47");
    expect(sparse_moe_candidate_supports_tokens(SparseMoeCandidate::Prefill, 1), "lab prefill rejected T=1");
    expect(sparse_moe_candidate_supports_tokens(SparseMoeCandidate::Decode, 4097), "column decode rejected a sliced extent");
    try {
        (void)parse_sparse_moe_candidate("small_t");
        expect(false, "unknown candidate silently fell back");
    } catch (const std::invalid_argument&) {}
    expect(!sparse_moe_candidate_supports_tokens(static_cast<SparseMoeCandidate>(255), 4),
           "invalid candidate admitted");
    expect(parse_sparse_moe_frontier("2") == 2, "lowest frontier rejected");
    expect(parse_sparse_moe_frontier("20") == 20, "existing Q8 frontier rejected");
    expect(parse_sparse_moe_frontier("47") == 47, "highest compiled frontier rejected");
    for (const char* bad : {"", "1", "48", "02", "20x", "-1"}) {
        expect(parse_sparse_moe_frontier(bad) == 0, "invalid frontier bypassed the compiled kernel domain");
    }
    expect(sparse_moe_policy_candidate_valid("sparse_moe/prefill/q8-q8", "47"), "Q8 policy rejected");
    expect(sparse_moe_policy_candidate_valid("sparse_moe/prefill-tiles/q4-q6", "narrow"), "tile policy rejected");
    expect(!sparse_moe_policy_candidate_valid("sparse_moe/prefill/q8-q8", "48"), "unsafe frontier admitted");
    expect(!sparse_moe_policy_candidate_valid("sparse_moe/prefill-tiles/nvfp4", "wide"), "fixed NVFP4 tile overridden");
    return failures == 0 ? 0 : 1;
}
