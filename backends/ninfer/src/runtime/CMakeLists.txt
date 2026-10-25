# The measured device route profiles ship inside the binary: device_profiles.json (ninfer-calibrate
# --print output of each part, merged) becomes a string in the generated source.
set(NINFER_DEVICE_PROFILES_FILE ${CMAKE_CURRENT_SOURCE_DIR}/engine/device_profiles.json)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${NINFER_DEVICE_PROFILES_FILE})
file(READ ${NINFER_DEVICE_PROFILES_FILE} _ninfer_profiles_rest)
# MSVC rejects a string literal over 16,380 bytes (C2026) and adjacent literals that join to over
# 65,535 (C1091), so the JSON goes in as an array of raw literals of at most 8 KiB, each ending on a
# line boundary, which the generated function joins once.
set(NINFER_DEVICE_PROFILES_PARTS "")
string(LENGTH "${_ninfer_profiles_rest}" _ninfer_profiles_length)
while(_ninfer_profiles_length GREATER 0)
  set(_ninfer_profiles_cut ${_ninfer_profiles_length})
  if(_ninfer_profiles_length GREATER 8192)
    string(SUBSTRING "${_ninfer_profiles_rest}" 0 8192 _ninfer_profiles_head)
    string(FIND "${_ninfer_profiles_head}" "\n" _ninfer_profiles_cut REVERSE)
    if(_ninfer_profiles_cut LESS 0)
      message(FATAL_ERROR "device_profiles.json has a line longer than 8 KiB")
    endif()
    math(EXPR _ninfer_profiles_cut "${_ninfer_profiles_cut} + 1")
  endif()
  string(SUBSTRING "${_ninfer_profiles_rest}" 0 ${_ninfer_profiles_cut} _ninfer_profiles_part)
  string(SUBSTRING "${_ninfer_profiles_rest}" ${_ninfer_profiles_cut} -1 _ninfer_profiles_rest)
  string(APPEND NINFER_DEVICE_PROFILES_PARTS
         "            R\"ninfer_json(${_ninfer_profiles_part})ninfer_json\",\n")
  string(LENGTH "${_ninfer_profiles_rest}" _ninfer_profiles_length)
endwhile()
configure_file(engine/device_profiles_builtin.cpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/engine/device_profiles_builtin.cpp @ONLY)

add_library(ninfer_runtime_support STATIC
  engine/admission_policy.cpp
  engine/context_cache/context_cost.cpp
  engine/context_cache/context_cost_defaults.cpp
  engine/device_profile.cpp
  engine/device_profile_build_id.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/engine/device_profiles_builtin.cpp
  engine/kv_capacity.cpp
  engine/public_types.cpp
  prefix_cache/block_hash.cpp
  prefix_cache/prefix_index.cpp
  prefix_cache/tap_planner.cpp
)
ninfer_internal_includes(ninfer_runtime_support)
target_link_libraries(ninfer_runtime_support PUBLIC ninfer_core ninfer_ops PRIVATE ninfer::json)
# Calibrated device profiles record the build id (engine/device_profile_build_id.cpp).
ninfer_stamp_build_id(ninfer_runtime_support)

# Public Engine and common contracts; model execution remains in the model layer.
add_library(ninfer_engine STATIC
  engine/model_instance.cpp
  engine/qwen4_exp_core.cpp
  contract/sampling.cpp
  engine/engine.cpp
)
ninfer_internal_includes(ninfer_engine)
target_include_directories(ninfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ninfer_engine PRIVATE
  ninfer_calibration
  ninfer_model_runtime
  ninfer_artifact
  ninfer_core
  ninfer_runtime_support
  ninfer_ops
  ninfer_text
  ninfer_media_decode)
add_library(ninfer::engine ALIAS ninfer_engine)
