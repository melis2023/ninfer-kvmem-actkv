# End-to-end product benchmark: public Engine API and native .ninfer artifacts only.
set(_ninfer_bench_sources
  "${CMAKE_CURRENT_LIST_DIR}/ninfer_bench.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ninfer_bench_support.cpp")
if(NINFER_SHARED_RUNTIME)
  ninfer_runtime_program(ninfer_bench ninfer_app_bench
    SOURCES ${_ninfer_bench_sources} LIBRARIES ninfer_engine ${NINFER_CUDART_TARGET})
  set(_ninfer_bench_target ninfer_app_bench)
else()
  add_executable(ninfer_bench ${_ninfer_bench_sources})
  target_link_libraries(ninfer_bench PRIVATE ninfer_engine ${NINFER_CUDART_TARGET})
  ninfer_stage_runtime_dlls(ninfer_bench)
  set(_ninfer_bench_target ninfer_bench)
endif()
ninfer_internal_includes(${_ninfer_bench_target})
target_include_directories(${_ninfer_bench_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(${_ninfer_bench_target} PRIVATE NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(${_ninfer_bench_target} PRIVATE kvmem)
