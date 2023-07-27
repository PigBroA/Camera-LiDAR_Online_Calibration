cmake_minimum_required(VERSION 3.6)
project(patchworkpp)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIRS}
)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(patchworkpp SHARED src/patchworkpp.cpp)

target_link_libraries(patchworkpp ${Eigen_LIBRARIES})