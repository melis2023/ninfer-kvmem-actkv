target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/frontend/frontend.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/output_session.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/chat_template.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/prompt_layout.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/digest.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/graft.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/processor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/media_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_call_parser.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_contract.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_grammar.cpp"
)
target_include_directories(ninfer_model_runtime PRIVATE ${PROJECT_SOURCE_DIR}/third_party/xgrammar/cpp)
target_link_libraries(ninfer_model_runtime PRIVATE ninfer_xgrammar)
