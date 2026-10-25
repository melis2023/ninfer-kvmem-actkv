ninfer_add_test(ninfer_admission_policy_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_admission_policy.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_context_cost_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_context_cost.cpp"
  LIBRARIES ninfer_runtime_support ninfer::json)

ninfer_add_test(ninfer_slot_spill_guard_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_slot_spill_guard.cpp")

ninfer_add_test(ninfer_resource_manager_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_resource_manager.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_kv_capacity_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_capacity.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_device_profile_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device_profile.cpp"
  LIBRARIES ninfer_runtime_support ninfer_calibration)

ninfer_add_test(ninfer_context_cache_defaults_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_context_cache_defaults.cpp"
  LIBRARIES ninfer_engine ninfer_core ninfer::json)

ninfer_add_test(ninfer_sampling_defaults_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_sampling_defaults.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_prefix_cache_index_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prefix_cache_index.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_effective_thinking_budget_test
  STANDALONE
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_effective_thinking_budget.cpp")

ninfer_add_test(ninfer_engine_logprobs_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_engine_logprobs_real.cpp"
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_engine_logprobs_real_test PROPERTIES SKIP_RETURN_CODE 77)
