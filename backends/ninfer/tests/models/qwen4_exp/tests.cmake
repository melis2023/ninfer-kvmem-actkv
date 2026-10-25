ninfer_add_test(ninfer_qwen4_exp_ngram_hash_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_hash.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen4_exp_ngram_table_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_table.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen4_exp_ngram_component_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_component.cpp"
  LIBRARIES ninfer_model_loading)

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
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_qwen4_exp_generate_real PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen4_exp_expert_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_cache.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_qwen4_exp_expert_cache_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen4_exp_expert_stream_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_stream.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_qwen4_exp_expert_stream_test PROPERTIES SKIP_RETURN_CODE 77)
