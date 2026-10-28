target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/page_softmax.cu")
# The scorer is a KVMem host-library client: it reads mean-K blocks from RawKvStore
# and owns only bounded device staging buffers.
target_link_libraries(ninfer_ops PUBLIC kvmem)
