ninfer_add_test(ninfer_flint_kernels_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kernels.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_flint_kernels_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_flint_branches_real_test STANDALONE
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_branches_real.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_flint_branches_real_test PROPERTIES SKIP_RETURN_CODE 77)
