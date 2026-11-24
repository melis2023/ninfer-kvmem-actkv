target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/trellis_reference.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/trellis_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/trellis_kernels.cu"
)
