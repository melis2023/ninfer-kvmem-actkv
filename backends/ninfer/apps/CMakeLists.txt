# Every program links the whole kernel image (~0.5 GB per architecture). NINFER_MULTICALL links the
# four into one executable, ninfer-multicall (cmake/NinferBundles.cmake), with each program name a
# symlink to it beside it: the container image carries the kernels once instead of four times.
option(NINFER_MULTICALL
       "Link ninfer, ninfer-serve, ninfer-calibrate and ninfer-perplexity into one multi-call executable"
       OFF)
if(NINFER_MULTICALL)
  if(WIN32)
    message(FATAL_ERROR "NINFER_MULTICALL names its programs with symlinks; it is for Linux")
  endif()
  include(${PROJECT_SOURCE_DIR}/cmake/NinferBundles.cmake)
endif()

# ninfer_app(<name> <bundle program> [BUILD_ID] SOURCES ... LIBRARIES ...): <name> as an executable,
# or with NINFER_MULTICALL as <bundle program> of ninfer-multicall, run under <name>.
function(ninfer_app name program)
  cmake_parse_arguments(PARSE_ARGV 2 arg "BUILD_ID" "" "SOURCES;LIBRARIES")
  if(NINFER_MULTICALL)
    ninfer_bundle_program(ninfer-multicall ${program} COMMAND ${name}
      SOURCES ${arg_SOURCES} LIBRARIES ${arg_LIBRARIES})
    set(target ${program})
  else()
    add_executable(${name} ${arg_SOURCES})
    target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
    ninfer_stage_runtime_dlls(${name})
    set(target ${name})
  endif()
  ninfer_internal_includes(${target})
  if(arg_BUILD_ID)
    ninfer_stamp_build_id(${target})
  endif()
endfunction()

ninfer_app(ninfer ninfer_app_cli BUILD_ID
  SOURCES cli/main.cpp cli/options.cpp
  LIBRARIES ninfer_engine ninfer_product_logging ninfer_product_prompt_input)
ninfer_app(ninfer-serve ninfer_app_serve BUILD_ID
  SOURCES serve/main.cpp
  LIBRARIES ninfer_serve ninfer_product_logging)
ninfer_app(ninfer-calibrate ninfer_app_calibrate
  SOURCES calibrate/main.cpp
  LIBRARIES ninfer_calibration ninfer_runtime_support)
ninfer_app(ninfer-perplexity ninfer_app_perplexity BUILD_ID
  SOURCES perplexity/main.cpp perplexity/corpus.cpp perplexity/evaluation.cpp
  LIBRARIES ninfer_engine ninfer_product_logging ninfer::json)

if(NINFER_MULTICALL)
  ninfer_finalize_bundle(ninfer-multicall)
  foreach(name IN ITEMS ninfer ninfer-serve ninfer-calibrate ninfer-perplexity)
    add_custom_command(TARGET ninfer-multicall POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E create_symlink ninfer-multicall
              $<TARGET_FILE_DIR:ninfer-multicall>/${name})
  endforeach()
endif()

if(WIN32 AND NINFER_DIRECTSTORAGE)
  foreach(app IN ITEMS ninfer ninfer-serve)
    add_custom_command(TARGET ${app} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NINFER_DIRECTSTORAGE_DLLS}
              $<TARGET_FILE_DIR:${app}>)
  endforeach()
endif()
