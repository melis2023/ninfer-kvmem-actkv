add_library(ninfer_core STATIC
  arena.cu
  resident_memory.cpp
  resident_scan.cu
  device.cu
  stage_link.cu
  tensor_parallel.cu
  dtype.cpp
  gpu_clock_floor.cpp
  host_worker_pool.cpp
  cpu_spin_pool.cpp
  expert_split.cpp
  expert_tier.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_kv_arena.cpp
  disk_kv_store.cpp
  disk_kv_bridge.cpp
  direct_storage_reader.cpp
  file_read_queue.cpp
  paged_kv_cache.cpp
  kv_expert_loan.cpp
  linear_attention_state.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  decode_graph.cpp
  cuda_host_handshake.cu
  expert_delivery.cu
  evictable_kv_pool.cu
  evictable_weight_pool.cu
  vmm.cu
  growable_region.cu
  device_snapshot.cu
)
ninfer_internal_includes(ninfer_core)
# The architectures this build carries, for the message that refuses a device it cannot run. A
# list's semicolons would split the property into several definitions.
string(REPLACE ";" " " _ninfer_cuda_architectures "${CMAKE_CUDA_ARCHITECTURES}")
set_source_files_properties(device.cu PROPERTIES
  COMPILE_DEFINITIONS "NINFER_CUDA_ARCHITECTURES=\"${_ninfer_cuda_architectures}\"")
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ${NINFER_CUDART_TARGET} CUDA::nvtx3 Threads::Threads CUDA::cuda_driver)
if(MSVC)
  # --help must run on build hosts without an NVIDIA driver. Resolve driver entry points only
  # when execution uses them; the CUDA runtime still checks device availability at startup.
  target_link_libraries(ninfer_core PUBLIC delayimp)
  target_link_options(ninfer_core INTERFACE "$<HOST_LINK:/DELAYLOAD:nvcuda.dll>")
endif()
# gpu_clock_floor.cpp loads NVML at run time.
target_link_libraries(ninfer_core PRIVATE ${CMAKE_DL_LIBS})
if(WIN32)
  target_link_libraries(ninfer_core PUBLIC pdh dxgi)
endif()
if(WIN32 AND NINFER_DIRECTSTORAGE)
  target_link_libraries(ninfer_core PUBLIC ninfer_directstorage)
endif()
