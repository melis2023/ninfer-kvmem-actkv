if(NINFER_WINDOWS_KERNEL_DLL)
  # Both the GGML bridge and Ops use one device-tier probe and cache in the DLL.
  add_library(ninfer_device_tier OBJECT "${CMAKE_CURRENT_LIST_DIR}/common/device_tier.cu")
  ninfer_internal_includes(ninfer_device_tier)
  ninfer_cuda_archive(ninfer_device_tier)
else()
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/common/device_tier.cu")
endif()

target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/weight_input.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/common/device_route.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/add_bias.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/argmax.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/cast.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/causal_conv1d.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/embed_gather.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/gdn_gating.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/gelu.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/hadamard_transform.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/l2norm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/layer_norm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/logprob_topk.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/mtp_pack.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/mtp_round.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/speculative_round.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/paged_kv_window.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/position.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/prepare_ragged_prefix.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/prepare_masked_block.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/residual_add.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/rmsnorm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/rmsnorm_pack_tail.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/rope.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/sampling.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/scalar.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/scatter.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/sigmoid_gate_mul.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/silu_and_mul.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/readout_logprobs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/target_logprobs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/vision_pos_embed.cu"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/add_bias.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/argmax.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/cast.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/causal_conv1d_silu.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/embedding.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/gdn_gating.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/gelu.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/hadamard_transform.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/l2norm.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/layer_norm.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/logprob_topk.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/mtp_pack.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/mtp_round.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/speculative_round.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/paged_kv_window.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/position.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/prepare_ragged_prefix.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/prepare_masked_block.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/residual_add.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/rmsnorm.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/rmsnorm_pack_tail.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/rope.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/sampling.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/scalar.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/scatter.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/sigmoid_mul.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/hyper_connection/hyper_connection.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ple_inject/ple_inject.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ple_inject/ngram_rows.cu"
  "${CMAKE_CURRENT_LIST_DIR}/qsa_indexer/qsa_indexer.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_attention/sparse_attention.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kv_stream/kv_stream.cu"
  "${CMAKE_CURRENT_LIST_DIR}/moe_route/moe_route.cu"
  "${CMAKE_CURRENT_LIST_DIR}/moe_experts/moe_experts_bf16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/moe_experts/moe_experts_gguf.cu"
  "${CMAKE_CURRENT_LIST_DIR}/moe_experts/moe_experts_native.cu"
  "${CMAKE_CURRENT_LIST_DIR}/moe_experts/moe_expert_cpu.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/moe_experts/gguf_expert_cpu.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/silu_mul.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/readout_logprobs.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/target_logprobs.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/vision_pos_embed.cpp"
)

# The GGUF CPU expert products read ggml's i-quant grids (ggml-common.h only) and reproduce the
# GPU's products bit for bit, so no compiler may fuse their multiplies and adds.
set(gguf_cpu_no_contract "")
if(NOT MSVC)
  set(gguf_cpu_no_contract "-ffp-contract=off")
endif()
set_source_files_properties(
  "${CMAKE_CURRENT_LIST_DIR}/moe_experts/gguf_expert_cpu.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/moe_experts/gguf_expert_cpu_avx2.cpp"
  PROPERTIES INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/third_party/ggml-quants/src")
set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/moe_experts/gguf_expert_cpu.cpp"
  PROPERTIES COMPILE_OPTIONS "${gguf_cpu_no_contract}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(cpu_expert_dir "${CMAKE_CURRENT_LIST_DIR}/moe_experts")
  target_sources(ninfer_ops PRIVATE
    "${cpu_expert_dir}/moe_expert_cpu_avx2.cpp"
    "${cpu_expert_dir}/moe_expert_cpu_avx512.cpp"
    "${cpu_expert_dir}/gguf_expert_cpu_avx2.cpp")
  set_source_files_properties("${cpu_expert_dir}/moe_expert_cpu.cpp" "${cpu_expert_dir}/gguf_expert_cpu.cpp"
    PROPERTIES COMPILE_DEFINITIONS NINFER_CPU_EXPERT_X86=1)
  if(MSVC)
    set_source_files_properties("${cpu_expert_dir}/moe_expert_cpu_avx2.cpp" "${cpu_expert_dir}/gguf_expert_cpu_avx2.cpp"
      PROPERTIES COMPILE_OPTIONS /arch:AVX2 COMPILE_DEFINITIONS NINFER_CPU_EXPERT_X86=1)
    set_source_files_properties("${cpu_expert_dir}/moe_expert_cpu_avx512.cpp"
      PROPERTIES COMPILE_OPTIONS /arch:AVX512 COMPILE_DEFINITIONS NINFER_CPU_EXPERT_X86=1)
  else()
    set_source_files_properties("${cpu_expert_dir}/moe_expert_cpu_avx2.cpp"
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c" COMPILE_DEFINITIONS NINFER_CPU_EXPERT_X86=1)
    set_source_files_properties("${cpu_expert_dir}/gguf_expert_cpu_avx2.cpp"
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off" COMPILE_DEFINITIONS NINFER_CPU_EXPERT_X86=1)
    set_source_files_properties("${cpu_expert_dir}/moe_expert_cpu_avx512.cpp"
      PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni;-mavx512dq;-mf16c;-mfma"
      COMPILE_DEFINITIONS NINFER_CPU_EXPERT_X86=1)
  endif()
endif()
