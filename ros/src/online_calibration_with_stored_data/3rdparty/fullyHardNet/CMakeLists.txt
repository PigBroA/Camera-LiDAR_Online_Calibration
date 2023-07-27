cmake_minimum_required(VERSION 3.6)
project(fullyHardNet)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(fullyHardNet SHARED src/fullyHardNet.cpp)

target_link_libraries(fullyHardNet ${TORCH_LIBRARIES})