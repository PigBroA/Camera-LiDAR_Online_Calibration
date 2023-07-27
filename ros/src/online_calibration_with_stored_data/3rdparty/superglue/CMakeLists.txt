cmake_minimum_required(VERSION 3.6)
project(superglue)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${TensorRT_INCLUDE_Dir}   
)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(superglue SHARED src/super_point.cpp src/super_glue.cpp)

target_link_libraries(superglue ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${Eigen_LIBRARIES} ${YAML_CPP_LIBRARIES} tensorrtbuffer)