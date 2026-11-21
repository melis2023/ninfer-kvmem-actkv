target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/model.cpp"
)
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/drafter.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/runtime.cpp"
)
