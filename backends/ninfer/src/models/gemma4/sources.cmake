target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/model.cpp"
)
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/cpu_experts.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/executor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kv_format.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kv_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/vision.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/vision_cpu.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/vision_cpu_kernels.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/vision_kernels.cu"
)

# The CPU vision tower's AVX2/FMA kernels, chosen at run time on CPUs that have them.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(gemma4_vision_cpu_avx2 "${CMAKE_CURRENT_LIST_DIR}/vision_cpu_kernels_avx2.cpp")
  target_sources(ninfer_model_runtime PRIVATE "${gemma4_vision_cpu_avx2}")
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/vision_cpu_kernels.cpp"
    PROPERTIES COMPILE_DEFINITIONS NINFER_VISION_CPU_X86=1)
  if(MSVC)
    set_source_files_properties("${gemma4_vision_cpu_avx2}"
      PROPERTIES COMPILE_OPTIONS /arch:AVX2 COMPILE_DEFINITIONS NINFER_VISION_CPU_X86=1)
  else()
    set_source_files_properties("${gemma4_vision_cpu_avx2}"
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma" COMPILE_DEFINITIONS NINFER_VISION_CPU_X86=1)
  endif()
endif()
