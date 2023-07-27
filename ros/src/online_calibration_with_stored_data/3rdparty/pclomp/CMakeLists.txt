cmake_minimum_required(VERSION 3.6)
project(pclomp)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PCL_INCLUDE_DIRS}
)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(pclomp SHARED src/voxel_grid_covariance_omp.cpp src/ndt_omp.cpp src/gicp_omp.cpp)

target_link_libraries(pclomp ${PCL_LIBRARIES})