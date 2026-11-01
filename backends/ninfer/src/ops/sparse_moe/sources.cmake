target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/decode/sparse_moe_decode_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/decode/sparse_moe_decode_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/prefill/sparse_moe_prefill_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/prefill/sparse_moe_prefill_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/small_t/sparse_moe_small_t_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/small_t/sparse_moe_small_t_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/streamed.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/streamed_cpu.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/sparse_moe.cpp"
)

# The streamed experts' CPU products reproduce the GPU's bit for bit: no compiler may fuse their
# multiplies and adds.
set(stream_cpu_options "")
if(NOT MSVC)
  set(stream_cpu_options "-ffp-contract=off")
endif()
set(stream_cpu_definitions "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(stream_cpu_definitions NINFER_CPU_EXPERT_X86=1)
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/streamed_cpu_avx2.cpp")
  if(MSVC)
    set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/streamed_cpu_avx2.cpp"
      PROPERTIES COMPILE_OPTIONS /arch:AVX2 COMPILE_DEFINITIONS NINFER_CPU_EXPERT_X86=1)
  else()
    set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/streamed_cpu_avx2.cpp"
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off"
      COMPILE_DEFINITIONS NINFER_CPU_EXPERT_X86=1)
  endif()
endif()
set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/streamed_cpu.cpp"
  PROPERTIES COMPILE_OPTIONS "${stream_cpu_options}" COMPILE_DEFINITIONS "${stream_cpu_definitions}")
