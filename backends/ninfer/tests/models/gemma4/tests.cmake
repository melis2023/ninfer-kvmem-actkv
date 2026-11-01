ninfer_add_test(ninfer_gemma4_kernels_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kernels.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_gemma4_kernels_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_gemma4_assistant_real_test STANDALONE
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_assistant_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer::json)
set_tests_properties(ninfer_gemma4_assistant_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_gemma4_residency_real_test STANDALONE
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_residency_real.cpp"
  LIBRARIES ninfer::engine ninfer::json)
set_tests_properties(ninfer_gemma4_residency_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_gemma4_vision_kernels_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_kernels.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_gemma4_vision_kernels_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_gemma4_vision_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer::json)
set_tests_properties(ninfer_gemma4_vision_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_gemma4_vision_cpu_kernels_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_cpu_kernels.cpp"
  LIBRARIES ninfer_model_runtime)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  target_compile_definitions(ninfer_gemma4_vision_cpu_kernels_test PRIVATE NINFER_VISION_CPU_X86=1)
endif()

ninfer_add_test(ninfer_gemma4_kv_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_gemma4_kv_test PROPERTIES SKIP_RETURN_CODE 77)
