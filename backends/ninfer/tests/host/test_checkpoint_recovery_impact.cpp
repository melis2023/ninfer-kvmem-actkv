#include "runtime/contract/resources.h"

#include <array>

using ninfer::runtime::CheckpointRecoveryAlternativeWork;
using ninfer::runtime::PressureCheckpointRecoveryImpact;

constexpr bool recovery_impacts_compare_values() {
    const std::array alternatives{
        CheckpointRecoveryAlternativeWork{.transfers = {}, .prefill = {.tokens = 10}},
        CheckpointRecoveryAlternativeWork{.transfers = {}, .prefill = {.tokens = 20}},
    };
    auto independent = alternatives;
    const PressureCheckpointRecoveryImpact left{
        .owner = {7}, .checkpoint = {.frontier = 42}, .target_recovery_work = alternatives};
    auto right = left;
    right.target_recovery_work = independent;
    if (left != right) { return false; }
    independent[0].prefill.tokens = 11;
    if (left == right) { return false; }
    independent = alternatives;
    independent[1].transfers[0].copy_operations = 1;
    if (left == right) { return false; }
    right = left;
    right.target_recovery_work = std::span(alternatives).first(1);
    if (left == right) { return false; }
    right = left;
    right.owner.value = 8;
    if (left == right) { return false; }
    right = left;
    right.checkpoint.frontier = 43;
    if (left == right) { return false; }
    right = left;
    right.survives = false;
    if (left == right) { return false; }
    return PressureCheckpointRecoveryImpact{} == PressureCheckpointRecoveryImpact{};
}

static_assert(recovery_impacts_compare_values());

int main() {
    return recovery_impacts_compare_values() ? 0 : 1;
}
