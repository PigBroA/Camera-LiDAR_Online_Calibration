cmake_minimum_required(VERSION 3.6)
project(bts)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(bts SHARED src/bts.cpp src/densenet.cpp)

target_link_libraries(bts ${TORCH_LIBRARIES})