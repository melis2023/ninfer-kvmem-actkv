# The portable launcher reads the same model catalog before it loads a CUDA backend.
add_library(ninfer_model_catalog STATIC model_catalog.cpp)
ninfer_internal_includes(ninfer_model_catalog)

add_library(ninfer_serve STATIC
  http_transport.cpp
  console_log.cpp
  request_validation.cpp
  structured_output.cpp
  openai_chat_request.cpp
  openai_chat_response.cpp
  openai_chat_http.cpp
  text_completion.cpp
  rerank.cpp
  systemone.cpp
  decisions_http.cpp
  llamacpp_http.cpp
  openai_common.cpp
  openai_responses_request.cpp
  openai_responses_state.cpp
  openai_responses_response.cpp
  anthropic_messages_request.cpp
  anthropic_messages_response.cpp
  anthropic_messages_http.cpp
  openai_responses_store.cpp
  translate.cpp
  generation_service.cpp
  branch_memo.cpp
  request_events.cpp
  console_stats.cpp
  operational_log.cpp
  request_log.cpp
  http_server.cpp
  mcp_proxy.cpp
  load_report.cpp
  serve_metrics.cpp
  openai_responses_http.cpp
  serve_options.cpp
  stop_control.cpp
  model_registry.cpp
)
ninfer_internal_includes(ninfer_serve)
target_link_libraries(ninfer_serve PUBLIC kvmem PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(ninfer_serve
  PUBLIC ninfer_engine ninfer_model_catalog Threads::Threads ninfer::json ninfer::httplib
  PRIVATE ninfer_media_acquire ninfer_product_logging ninfer_product_prompt_input
          ${NINFER_CUDART_TARGET})

# A WebUI is compiled in only from a local directory, never downloaded at configure time, so offline
# and hermetic builds are unaffected. scripts/fetch-webui.sh unpacks llama.cpp's released WebUI into
# such a directory.
set(NINFER_WEBUI_DIR "" CACHE PATH
    "Directory of built WebUI files (with index.html) to serve from ninfer-serve; empty embeds none")

# A build directory that inherits no cache would otherwise compile webui_none.cpp and ship a
# ninfer-serve whose GET / is a bare 404 -- the build log never says the UI was skipped. Pick up
# the bundle this checkout already assembled instead, and say so.
if(NOT NINFER_WEBUI_DIR)
  foreach(NINFER_WEBUI_CANDIDATE
          "${CMAKE_SOURCE_DIR}/../../build-ui/full/dist"
          "${CMAKE_SOURCE_DIR}/../../build/share/kvmem/ui")
    if(EXISTS "${NINFER_WEBUI_CANDIDATE}/index.html" OR
       EXISTS "${NINFER_WEBUI_CANDIDATE}/index.html.gz")
      set(NINFER_WEBUI_DIR "${NINFER_WEBUI_CANDIDATE}" CACHE PATH
          "Directory of built WebUI files (with index.html) to serve from ninfer-serve; empty embeds none"
          FORCE)
      break()
    endif()
  endforeach()
endif()

if(NINFER_WEBUI_DIR)
  if(NOT EXISTS "${NINFER_WEBUI_DIR}/index.html" AND NOT EXISTS "${NINFER_WEBUI_DIR}/index.html.gz")
    message(FATAL_ERROR "NINFER_WEBUI_DIR has no index.html: ${NINFER_WEBUI_DIR}")
  endif()
  message(STATUS "WebUI: embedding ${NINFER_WEBUI_DIR}")
  add_executable(ninfer_webui_embed webui/embed.cpp)
  file(GLOB_RECURSE NINFER_WEBUI_FILES CONFIGURE_DEPENDS "${NINFER_WEBUI_DIR}/*")
  set(NINFER_WEBUI_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/webui_assets.cpp")
  add_custom_command(
    OUTPUT "${NINFER_WEBUI_SOURCE}"
    COMMAND ninfer_webui_embed "${NINFER_WEBUI_SOURCE}" "${NINFER_WEBUI_DIR}"
    DEPENDS ninfer_webui_embed ${NINFER_WEBUI_FILES}
    COMMENT "Embedding the WebUI from ${NINFER_WEBUI_DIR}"
    VERBATIM)
  target_sources(ninfer_serve PRIVATE "${NINFER_WEBUI_SOURCE}")
else()
  target_sources(ninfer_serve PRIVATE webui_none.cpp)
  message(WARNING
    "WebUI: NINFER_WEBUI_DIR is empty, so ninfer-serve is built without a WebUI and GET / "
    "answers 404. Pass -DNINFER_WEBUI_DIR=<dir with index.html> (build-ui/full/dist in this "
    "checkout) to embed one; --no-webui only hides an embedded UI at run time.")
endif()
