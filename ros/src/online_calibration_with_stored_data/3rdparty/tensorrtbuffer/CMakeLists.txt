cmake_minimum_required(VERSION 3.6)
project(tensorrtbuffer)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_Dir}
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(tensorrtbuffer SHARED src/logger.cpp)

target_link_libraries(tensorrtbuffer ${CUDA_LIBRARIES} ${TensorRT_LIB_Dir}/libnvparsers.so ${TensorRT_LIB_Dir}/libnvinfer.so ${TensorRT_LIB_Dir}/libnvonnxparser.so)