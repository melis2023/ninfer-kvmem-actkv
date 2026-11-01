find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/NinferTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/RuntimeTests.cmake)
include(cmake/ProductTests.cmake)
include(cmake/BenchSupportTests.cmake)
include(models/qwen3_5/tests.cmake)
include(models/qwen4_exp/tests.cmake)
include(models/gemma4/tests.cmake)
include(models/flint/tests.cmake)
ninfer_add_test(ninfer_tool_constraints_test STANDALONE
  SOURCES models/qwen3_5/test_tool_constraints.cpp LIBRARIES ninfer_serve ninfer_model_runtime ninfer_text)
ninfer_stage_runtime_dlls(ninfer_tool_constraints_test)
include(ops/tests.cmake)

ninfer_finalize_bundle(ninfer_tests)

set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_dflash2_tests_PROGRAMS
  ninfer_sliding_window_attention_test ninfer_qwen3_5_state_image_test
  ninfer_qwen3_5_state_image_layout_test)
ninfer_finalize_bundle(ninfer_dflash2_tests)
ninfer_stage_runtime_dlls(ninfer_dflash2_tests)

# Draft format composition and registered NVFP4 geometry oracles.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_dflash2_format_tests_PROGRAMS
  ninfer_linear_dynamic_grouped_conv_add_test ninfer_linear_nvfp4_a16_test)
ninfer_finalize_bundle(ninfer_dflash2_format_tests)
ninfer_stage_runtime_dlls(ninfer_dflash2_format_tests)

# Reuse only the objects needed to validate the native memory-policy contracts and
# the C++20 unique-string compatibility fix; do not link the full numerical suite.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_memory_policy_tests_PROGRAMS
  ninfer_cli_options_test ninfer_serve_options_test ninfer_kv_capacity_test
  ninfer_context_cache_defaults_test ninfer_unique_strings_test)
ninfer_finalize_bundle(ninfer_memory_policy_tests)
ninfer_stage_runtime_dlls(ninfer_memory_policy_tests)

# Native prefix admission must agree with the actual Device/Host restore path.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_prefix_cache_tests_PROGRAMS
  ninfer_qwen3_5_hybrid_prefix_real_test)
ninfer_finalize_bundle(ninfer_prefix_cache_tests)
ninfer_stage_runtime_dlls(ninfer_prefix_cache_tests)
add_test(NAME ninfer_hybrid_prefix_admission_test
  COMMAND ninfer_prefix_cache_tests ninfer_qwen3_5_hybrid_prefix_real_test)
set_tests_properties(ninfer_hybrid_prefix_admission_test PROPERTIES
  ENVIRONMENT "NINFER_HYBRID_REAL_SCENARIO=restore-admission" SKIP_RETURN_CODE 77)

# Focused Query/Pressure frontend and replay-budget option contracts.
ninfer_add_test(ninfer_kvmem_checkpoint_pool_test
  SOURCES models/qwen3_5/test_checkpoint_buffer_pool.cpp LIBRARIES ninfer_core ninfer_model_runtime)
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_query_policy_tests_PROGRAMS
  ninfer_serve_options_test ninfer_qwen3_5_frontend_test ninfer_kvmem_checkpoint_pool_test)
ninfer_finalize_bundle(ninfer_query_policy_tests)
ninfer_stage_runtime_dlls(ninfer_query_policy_tests)
add_test(NAME ninfer_kvmem_checkpoint_metadata_test
  COMMAND ninfer_query_policy_tests ninfer_kvmem_checkpoint_pool_test --metadata-only)

# Request accounting and protocol renderers, without unrelated GPU numerical tests.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_request_accounting_tests_PROGRAMS
  ninfer_request_log_test ninfer_console_stats_test
  ninfer_openai_schema_test ninfer_text_completion_test)
ninfer_finalize_bundle(ninfer_request_accounting_tests)
ninfer_stage_runtime_dlls(ninfer_request_accounting_tests)

# Shape-generalization regressions without linking unrelated model test families.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_dense_family_tests_PROGRAMS
  ninfer_gdn_gating_proj_test ninfer_linear_gguf_test ninfer_gdn_replay_fold_test
  ninfer_softmax_attention_test)
ninfer_finalize_bundle(ninfer_dense_family_tests)
ninfer_stage_runtime_dlls(ninfer_dense_family_tests)
add_test(NAME ninfer_dense_family_attention_test
  COMMAND ninfer_dense_family_tests ninfer_softmax_attention_test --dense-family-only)
add_test(NAME ninfer_dense_family_replay_test
  COMMAND ninfer_dense_family_tests ninfer_gdn_replay_fold_test --dense-family-only)

# Exit code 77 is a skip for every test (tests/cuda_availability.h). Without a usable CUDA device --
# no GPU, no driver, or only the toolkit's stub driver, as on a CI runner -- a test that has no check
# of its own fails on its first CUDA call with one of these runtime messages; report that as skipped
# too, so such a box still runs and judges every host test. None of the messages can occur where a
# device is usable.
get_property(_ninfer_all_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_ninfer_all_tests} PROPERTIES SKIP_RETURN_CODE 77 SKIP_REGULAR_EXPRESSION
  "CUDA driver is a stub library;cudaErrorStubLibrary;no CUDA-capable device is detected;CUDA driver version is insufficient for CUDA runtime version")
# Focused physical-pipeline qualification, reusing the existing program objects.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_pipeline_tests_PROGRAMS
  ninfer_multi_gpu_test ninfer_kv_capacity_test ninfer_serve_options_test
  ninfer_qwen3_5_stages_real_test)
set_property(GLOBAL APPEND PROPERTY NINFER_BUNDLE_ninfer_pipeline_tests_PROGRAMS
  ninfer_qwen3_5_loading_test)
set_property(GLOBAL APPEND PROPERTY NINFER_BUNDLE_ninfer_pipeline_tests_PROGRAMS
  ninfer_linear_gguf_test)
ninfer_finalize_bundle(ninfer_pipeline_tests)
ninfer_stage_runtime_dlls(ninfer_pipeline_tests)

# Every test executable shares one per-config runtime directory; stage the FFmpeg/curl DLL closure
# there once so they resolve their imports (the Windows loader only searches the executable's own
# directory, the system directories and PATH).
ninfer_stage_test_runtime_dlls(ninfer_tests)

# P1 feasibility probes: opt-in targets in the isolated checkout, no production CLI.
add_executable(ninfer_p1_engine experimental/p1_engine.cpp)
ninfer_test_includes(ninfer_p1_engine)
target_link_libraries(ninfer_p1_engine PRIVATE ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_p1_engine)
ninfer_add_test(ninfer_p1_sparse STANDALONE
  SOURCES experimental/p1_sparse.cpp LIBRARIES ninfer_ops)
ninfer_op_oracle_options(ninfer_p1_sparse)
ninfer_add_test(ninfer_p1_payload STANDALONE
  SOURCES experimental/p1_payload.cpp LIBRARIES ninfer_core)
ninfer_add_test(ninfer_p1_options STANDALONE
  SOURCES experimental/p1_options.cpp LIBRARIES ninfer::engine ninfer_core ninfer::json)
ninfer_stage_runtime_dlls(ninfer_p1_options)
ninfer_add_test(ninfer_p1_rope STANDALONE
  SOURCES experimental/p1_rope.cpp LIBRARIES ninfer_ops)
ninfer_op_oracle_options(ninfer_p1_rope)

# The query provenance tests are also runnable without linking unrelated GPU test families.
ninfer_add_test(ninfer_memory_frontend STANDALONE NEEDS_SOURCE_DIR
  SOURCES models/qwen3_5/test_frontend.cpp LIBRARIES ninfer_engine ninfer_core ninfer::json)
ninfer_stage_runtime_dlls(ninfer_memory_frontend)
ninfer_add_test(ninfer_token_sums STANDALONE
  SOURCES experimental/token_sums.cpp LIBRARIES ninfer_ops)
ninfer_op_oracle_options(ninfer_token_sums)
ninfer_add_test(ninfer_attention_page_mass STANDALONE
  SOURCES experimental/attention_page_mass.cpp LIBRARIES ninfer_ops)
ninfer_op_oracle_options(ninfer_attention_page_mass)
ninfer_add_test(ninfer_p4_retrieval STANDALONE
  SOURCES experimental/p4_retrieval.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_p4_retrieval)
ninfer_add_test(ninfer_p4_history STANDALONE
  SOURCES experimental/p4_history.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_p4_history)
ninfer_add_test(ninfer_memory_sessions STANDALONE
  SOURCES experimental/memory_sessions.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_sessions)
ninfer_add_test(ninfer_memory_ram_budget STANDALONE
  SOURCES experimental/memory_ram_budget.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_ram_budget)
ninfer_add_test(ninfer_active_kv_ssd STANDALONE
  SOURCES experimental/active_kv_ssd.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_active_kv_ssd)
ninfer_add_test(ninfer_ssd_failure STANDALONE
  SOURCES experimental/ssd_failure.cpp LIBRARIES ninfer::engine kvmem)
ninfer_stage_runtime_dlls(ninfer_ssd_failure)
ninfer_add_test(ninfer_memory_concurrency STANDALONE
  SOURCES experimental/memory_concurrency.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_concurrency)
ninfer_add_test(ninfer_migration_concurrency STANDALONE
  SOURCES experimental/migration_concurrency.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_migration_concurrency)
ninfer_add_test(ninfer_memory_vision STANDALONE
  SOURCES experimental/memory_vision.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_vision)
ninfer_add_test(ninfer_memory_disk STANDALONE
  SOURCES experimental/memory_disk.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_disk)
# Independently linked so the preceding P6-P9 qualification keeps its binaries.
ninfer_add_test(ninfer_memory_formats STANDALONE
  SOURCES experimental/p4_history.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_formats)
ninfer_add_test(ninfer_memory_ngram STANDALONE
  SOURCES experimental/memory_ngram.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_ngram)
ninfer_add_test(ninfer_p4_quality STANDALONE
  SOURCES experimental/p4_quality.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_p4_quality)

set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_kvmem_product_tests_PROGRAMS
  ninfer_serve_options_test ninfer_openai_schema_test ninfer_openai_responses_test
  ninfer_openai_responses_store_test ninfer_load_report_test ninfer_resource_manager_test
  ninfer_anthropic_schema_test)
ninfer_finalize_bundle(ninfer_kvmem_product_tests)
ninfer_stage_runtime_dlls(ninfer_kvmem_product_tests)
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_refresh_contract_tests_PROGRAMS
  ninfer_bench_support_test ninfer_tool_call_parser_test ninfer_structured_output_test
  ninfer_resource_manager_test ninfer_qwen3_5_frontend_test ninfer_speculative_round_test
  ninfer_qwen3_5_structured_round_test)
ninfer_finalize_bundle(ninfer_refresh_contract_tests)
ninfer_stage_runtime_dlls(ninfer_refresh_contract_tests)
add_test(NAME ninfer_openai_image_detail_chat_test
  COMMAND ninfer_kvmem_product_tests ninfer_openai_schema_test --image-detail-only)
add_test(NAME ninfer_openai_image_detail_responses_test
  COMMAND ninfer_kvmem_product_tests ninfer_openai_responses_test --image-detail-only)

# Reuse only the GGUF numerical test object for the focused format qualification.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_migration_gguf_tests_PROGRAMS
  ninfer_linear_gguf_test)
ninfer_finalize_bundle(ninfer_migration_gguf_tests)
ninfer_stage_runtime_dlls(ninfer_migration_gguf_tests)

# Keep codec, Attention and MTP route qualification runnable without linking
# unrelated numerical families into the NVFP4 integration executable.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_kvmem_nvfp4_tests_PROGRAMS
  ninfer_kv_cache_append_test ninfer_softmax_attention_test
  ninfer_qwen3_5_mtp_graph_profiles_test ninfer_serve_options_test)
ninfer_finalize_bundle(ninfer_kvmem_nvfp4_tests)
ninfer_stage_runtime_dlls(ninfer_kvmem_nvfp4_tests)

# NVFP4 integration checks use separate processes for startup-fixed scoped overrides.
ninfer_add_op_test(ninfer_nvfp4_scope STANDALONE
  SOURCES experimental/nvfp4_scope.cpp LIBRARIES ninfer_ops)
ninfer_stage_runtime_dlls(ninfer_nvfp4_scope)
foreach(route_scope IN ITEMS default int8:legacy int8:unified a16:legacy a16:unified)
  string(REPLACE ":" "_" route_name "${route_scope}")
  set(route_environment "${route_scope}")
  if(route_scope STREQUAL "default")
    set(route_environment "")
  endif()
  add_test(NAME ninfer_nvfp4_routing_${route_name} COMMAND ninfer_nvfp4_scope)
  set_tests_properties(ninfer_nvfp4_routing_${route_name} PROPERTIES
    SKIP_RETURN_CODE 77 ENVIRONMENT "NINFER_LINEAR_ROUTES=${route_environment}")
endforeach()
ninfer_add_test(ninfer_nvfp4_mode_binding STANDALONE
  SOURCES experimental/nvfp4_mode_binding.cpp LIBRARIES ninfer::engine)
set_tests_properties(ninfer_nvfp4_mode_binding PROPERTIES SKIP_RETURN_CODE 77)
ninfer_stage_runtime_dlls(ninfer_nvfp4_mode_binding)

# Reuse the five relevant oracle objects without building unrelated test families.
ninfer_add_op_test(ninfer_nvfp4_native_scale_test
  SOURCES experimental/nvfp4_native_scale.cu LIBRARIES ninfer_ops)
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_nvfp4_fork_tests_PROGRAMS
  ninfer_linear_nvfp4_a16_test ninfer_linear_nvfp4_int8_test
  ninfer_linear_nvfp4_int8_fused_norm_test ninfer_linear_swiglu_nvfp4_test
  ninfer_linear_add_nvfp4_test ninfer_nvfp4_native_scale_test)
ninfer_finalize_bundle(ninfer_nvfp4_fork_tests)
ninfer_stage_runtime_dlls(ninfer_nvfp4_fork_tests)

# Focused CPU Vision lifecycle and existing numerical/options/loading qualification.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_cpu_vision_tests_PROGRAMS
  ninfer_qwen3_5_vision_cpu_test ninfer_cli_options_test ninfer_serve_options_test
  ninfer_qwen3_5_loading_test ninfer_request_log_test)
ninfer_finalize_bundle(ninfer_cpu_vision_tests)
ninfer_stage_runtime_dlls(ninfer_cpu_vision_tests)

# Native PTQ1 qualification reuses the public test objects without linking every numerical suite.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_ptq1_tests_PROGRAMS
  ninfer_artifact_ptq1_test ninfer_linear_ptq1_test ninfer_linear_ptq1_a8_test
  ninfer_linear_swiglu_ptq1_test ninfer_linear_add_ptq1_test
  ninfer_attn_input_proj_test ninfer_gdn_input_proj_test ninfer_hadamard_transform_test
  ninfer_linear_topk_test ninfer_linear_t2_a16_test ninfer_linear_t2_a8_test)
ninfer_finalize_bundle(ninfer_ptq1_tests)
ninfer_stage_runtime_dlls(ninfer_ptq1_tests)

# Focused serving contract bundle; does not pull in GPU numerical tests.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_monitor_tests_PROGRAMS ninfer_openai_schema_test)
ninfer_finalize_bundle(ninfer_monitor_tests)
ninfer_stage_runtime_dlls(ninfer_monitor_tests)
