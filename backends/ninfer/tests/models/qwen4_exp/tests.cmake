ninfer_add_test(ninfer_qwen4_exp_ngram_hash_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_hash.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen4_exp_ngram_table_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_table.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen4_exp_ngram_component_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_component.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops)

add_test(NAME ninfer_qwen4_exp_ngram_writer_interop_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/ngram_writer_interop.py"
    $<TARGET_FILE:ninfer_tests>)
set_tests_properties(ninfer_qwen4_exp_ngram_writer_interop_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen4_exp_config_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_config.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen4_exp_slice_real
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_slice_real.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops)
set_tests_properties(ninfer_qwen4_exp_slice_real PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen4_exp_generate_real
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_generate_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer::json)
set_tests_properties(ninfer_qwen4_exp_generate_real PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen4_exp_engine_real
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_real.cpp"
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen4_exp_engine_real PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen4_exp_long_context_real
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_long_context_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)
set_tests_properties(ninfer_qwen4_exp_long_context_real PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen4_exp_expert_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_cache.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_qwen4_exp_expert_cache_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_expert_profile_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_expert_profile.cpp"
  LIBRARIES ninfer_model_runtime)

ninfer_add_test(ninfer_qwen4_exp_ngram_draft_prefetch_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_draft_prefetch.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_qwen4_exp_ngram_draft_prefetch_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)

ninfer_add_op_test(ninfer_qwen4_exp_hybrid_experts_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_hybrid_experts.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_qwen4_exp_hybrid_experts_test PROPERTIES
  SKIP_RETURN_CODE 77 TIMEOUT 180 ENVIRONMENT "CUDA_MODULE_LOADING=LAZY")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_library(ninfer_delayed_peer_copy SHARED "${CMAKE_CURRENT_LIST_DIR}/delayed_peer_copy.cpp")
  target_link_libraries(ninfer_delayed_peer_copy PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
  add_test(NAME ninfer_qwen4_exp_peer_copy_order_test
    COMMAND ${CMAKE_COMMAND} -E env "LD_PRELOAD=$<TARGET_FILE:ninfer_delayed_peer_copy>"
      "NINFER_FLASH_NEXT_SAMPLING_ONLY=1"
      $<TARGET_FILE:ninfer_tests> ninfer_qwen4_exp_engine_real)
  set_tests_properties(ninfer_qwen4_exp_peer_copy_order_test PROPERTIES SKIP_RETURN_CODE 77)
  add_library(ninfer_delayed_memset SHARED "${CMAKE_CURRENT_LIST_DIR}/delayed_memset.cpp")
  target_link_libraries(ninfer_delayed_memset PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
  add_test(NAME ninfer_qwen4_exp_expert_cache_visibility_test
    COMMAND ${CMAKE_COMMAND} -E env "LD_PRELOAD=$<TARGET_FILE:ninfer_delayed_memset>"
      $<TARGET_FILE:ninfer_tests> ninfer_qwen4_exp_expert_cache_test)
  set_tests_properties(ninfer_qwen4_exp_expert_cache_visibility_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

ninfer_add_test(ninfer_qwen4_exp_expert_stream_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_stream.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_qwen4_exp_expert_stream_test PROPERTIES SKIP_RETURN_CODE 77)
