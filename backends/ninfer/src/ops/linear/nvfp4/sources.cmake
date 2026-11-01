target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/shapes/dflash2.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/dflash2_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_format.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_int8_grouped.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_int8_fused_norm_quant.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_w4a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_dispatch_unified.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5120_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n8192_k4096_unified.cu"
)

if(TARGET ninfer_nvfp4_non_rdc)
  target_sources(ninfer_nvfp4_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4_tma.cu"
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_w4a4_tma.cu"
  )
endif()
