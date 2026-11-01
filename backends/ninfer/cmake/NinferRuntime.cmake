include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/NinferBundles.cmake)

# Private command entries keep C++ objects and their allocation inside one runtime.
function(ninfer_runtime_program name program)
  cmake_parse_arguments(PARSE_ARGV 2 arg "" "" "SOURCES;LIBRARIES")
  ninfer_bundle_program(ninfer_runtime ${program} COMMAND ${name}
    SOURCES ${arg_SOURCES} LIBRARIES ${arg_LIBRARIES})
  add_executable(${name} "${PROJECT_SOURCE_DIR}/apps/launcher/main.cpp")
  target_compile_definitions(${name} PRIVATE NINFER_COMMAND="${name}")
  target_link_libraries(${name} PRIVATE ${CMAKE_DL_LIBS})
  if(name STREQUAL "ninfer-serve")
    target_compile_definitions(${name} PRIVATE NINFER_ROUTER_LAUNCHER=1)
    target_include_directories(${name} PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(${name} PRIVATE ninfer_model_catalog)
  endif()
  ninfer_stamp_build_id(${name})
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/portable/$<0:>")
  add_dependencies(${name} ninfer_runtime)
endfunction()

function(ninfer_finalize_runtime)
  get_property(programs GLOBAL PROPERTY NINFER_BUNDLE_ninfer_runtime_PROGRAMS)
  set(declarations "")
  set(entries "")
  set(libraries "")
  foreach(program IN LISTS programs)
    get_target_property(command ${program} NINFER_BUNDLE_COMMAND)
    string(APPEND declarations "int ninfer_bundle_entry_${program}(int, char**);\n")
    string(APPEND entries "    if (std::strcmp(command, \"${command}\") == 0) return ninfer_bundle_entry_${program}(argc, argv);\n")
    get_target_property(deps ${program} LINK_LIBRARIES)
    if(deps)
      list(APPEND libraries ${deps})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES libraries)
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/runtime_dispatch.cpp.in"
    "${CMAKE_BINARY_DIR}/generated/runtime_dispatch.cpp" @ONLY)
  add_library(ninfer_runtime SHARED "${CMAKE_BINARY_DIR}/generated/runtime_dispatch.cpp")
  target_link_libraries(ninfer_runtime PRIVATE ${programs} ${libraries})
  ninfer_stamp_build_id(ninfer_runtime)
  string(REGEX MATCH "^[0-9]+" cuda_major "${CMAKE_CUDA_COMPILER_VERSION}")
  set_target_properties(ninfer_runtime PROPERTIES OUTPUT_NAME ninfer-runtime
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/portable/backends/cuda${cuda_major}/$<0:>"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/portable/backends/cuda${cuda_major}/$<0:>"
    BUILD_RPATH_USE_ORIGIN ON)
  string(REPLACE ";" " " runtime_architectures "${CMAKE_CUDA_ARCHITECTURES}")
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/portable/backends/cuda${cuda_major}/architectures.txt"
    CONTENT "${runtime_architectures}\n")
  if(UNIX AND NOT APPLE)
    target_link_options(ninfer_runtime PRIVATE "LINKER:--exclude-libs,ALL")
  endif()
  ninfer_stage_runtime_dlls(ninfer_runtime)
  if(WIN32 AND NINFER_DIRECTSTORAGE)
    add_custom_command(TARGET ninfer_runtime POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NINFER_DIRECTSTORAGE_DLLS}
              $<TARGET_FILE_DIR:ninfer_runtime>)
  endif()
endfunction()
