set(CMAKE_CXX_STANDARD 17)

add_executable(index_hnsw HNSW/hnsw_index.cpp)
add_executable(search_hnsw HNSW/hnsw_search.cpp)

add_executable(rabit_index RaBitQ/index.cpp)
add_executable(rabit_fast_scan_search RaBitQ/search.cpp)
target_compile_definitions(rabit_fast_scan_search PRIVATE FAST_SCAN)
add_executable(rabit_fast_scan_search_rerank RaBitQ/search.cpp)
target_compile_definitions(rabit_fast_scan_search_rerank PRIVATE FAST_SCAN ESTIMATED_RERANK)


add_executable(res_index ResBitQ/res_index.cpp)
add_executable(res_split_index ResBitQ/res_index.cpp)
target_compile_definitions(res_split_index PRIVATE RESIDUAL_SPLIT)

add_executable(res_search ResBitQ/res_search.cpp)
add_executable(res_split_search ResBitQ/res_search.cpp)
target_compile_definitions(res_split_search PRIVATE RESIDUAL_SPLIT)

add_library(mrq_p1_bridge SHARED p1_bridge.cpp)
set_target_properties(
    mrq_p1_bridge
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)
