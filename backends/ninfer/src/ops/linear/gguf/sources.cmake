# ggml's block-quant kernels (third_party/ggml-quants) behind a plain bridge. They build as their own
# archive: the ggml headers define CUDA_CHECK and friends, so no NInfer translation unit may see them.
set(NINFER_GGML_QUANTS_ROOT "${PROJECT_SOURCE_DIR}/third_party/ggml-quants")
set(_ninfer_ggml_library_type STATIC)
if(NINFER_WINDOWS_KERNEL_DLL)
  set(_ninfer_ggml_library_type OBJECT)
endif()
add_library(ninfer_ggml_quants ${_ninfer_ggml_library_type}
  "${CMAKE_CURRENT_LIST_DIR}/ggml_bridge.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_fattn.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq1_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq2_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq2_xs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq2_xxs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq3_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq3_xxs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq4_nl.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq4_xs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q2_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q3_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q4_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q5_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q6_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q8_0.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q4_0.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q5_0.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q2_0.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q8_0.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_0.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q5_0.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q2_0.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q2_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q3_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q5_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q6_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq2_xxs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq2_xs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq2_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq3_xxs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq3_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq1_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq1_m.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq4_nl.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq4_xs.cu")
if(WIN32)
  # STATIC defaults to this name, but OBJECT defaults to a bare /Fd directory. Keep cache keys equal.
  set_target_properties(ninfer_ggml_quants PROPERTIES
    COMPILE_PDB_NAME ninfer_ggml_quants
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ninfer_ggml_quants.dir")
endif()
target_include_directories(ninfer_ggml_quants PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}"
  "${NINFER_GGML_QUANTS_ROOT}/include"
  "${NINFER_GGML_QUANTS_ROOT}/src")
ninfer_cuda_non_rdc_archive(ninfer_ggml_quants)
target_compile_options(ninfer_ggml_quants PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=177>)
target_link_libraries(ninfer_ggml_quants PRIVATE ${NINFER_CUDART_TARGET} CUDA::cublas)
if(NINFER_WINDOWS_KERNEL_DLL)
  # Compile independently of DLL-only definitions, preserving the static archive's cache identity.
  # Only linking changes. The bridge imports no global data or virtual tables.
  add_library(ninfer_ggml_quants_dll SHARED
    $<TARGET_OBJECTS:ninfer_ggml_quants> $<TARGET_OBJECTS:ninfer_device_tier>)
  set_target_properties(ninfer_ggml_quants_dll PROPERTIES
    OUTPUT_NAME ninfer_ggml_quants WINDOWS_EXPORT_ALL_SYMBOLS ON
    LINKER_LANGUAGE CUDA CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION OFF CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  target_link_libraries(ninfer_ggml_quants_dll PRIVATE ${NINFER_CUDART_TARGET} CUDA::cublas)
  target_link_libraries(ninfer_ops PRIVATE ninfer_ggml_quants_dll)
else()
  target_link_libraries(ninfer_ops PRIVATE ninfer_ggml_quants)
  # The flash attention's launch reads the device tier ninfer_ops keeps: a program whose other
  # objects never reach it still links (a link-only cycle the static archives resolve).
  target_link_libraries(ninfer_ggml_quants PRIVATE $<LINK_ONLY:ninfer_ops>)
endif()
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/gguf_linear.cpp")
