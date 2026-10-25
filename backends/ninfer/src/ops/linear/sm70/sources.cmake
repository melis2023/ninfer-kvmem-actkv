target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/sm70_dequantize.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sm70_gemm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sm70_integer.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sm70_project.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sm70_small.cu"
)
