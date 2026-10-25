target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/model.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ngram_component.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ngram_hash.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ngram_table.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/read_pool.cpp"
)
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/executor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_stream.cpp"
)
