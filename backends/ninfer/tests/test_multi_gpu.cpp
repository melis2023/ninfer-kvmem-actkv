// The pieces of the multi-GPU pipeline that need no model, in one executable: the stage plan and its
// solver, the KV pool and block tables across ranks, the boundary transfer between stages and the
// tensor-parallel exchange. The GPU ones run with two to four ranks on device 0 (the exchange also
// over two devices where a host has them) and report SKIP when there is no CUDA device.

int run_stage_plan_test();
int run_kv_cache_ranks_test();
int run_stage_link_test();
int run_tensor_parallel_test();

namespace {

// 77 is a suite that found no CUDA device; the others still run.
int failed(int status) { return status != 0 && status != 77 ? 1 : 0; }

} // namespace

int main() {
    int failures = failed(run_stage_plan_test());
    failures += failed(run_kv_cache_ranks_test());
    failures += failed(run_stage_link_test());
    failures += failed(run_tensor_parallel_test());
    return failures == 0 ? 0 : 1;
}
