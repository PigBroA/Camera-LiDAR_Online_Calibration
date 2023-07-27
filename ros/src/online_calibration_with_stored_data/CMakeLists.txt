cmake_minimum_required(VERSION 3.0.2)
project(online_calibration_with_stored_data)

add_compile_options(-std=c++17)

add_definitions(-DCMAKE_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}\")

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geodesy
  geometry_msgs
  pcl_ros
  roscpp
  sensor_msgs
)

set(CMAKE_PREFIX_PATH "/home/ldk/libtorch")
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(TensorRT_Dir "/home/ldk/TensorRT-8.2.3.0")
set(TensorRT_LIB_Dir "${TensorRT_Dir}/targets/x86_64-linux-gnu/lib")
set(TensorRT_INCLUDE_Dir "${TensorRT_Dir}/include/")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/fullyHardNet)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/bts)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tensorrtbuffer)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/superglue)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/calibration_dk)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glob)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/patchworkpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pclomp)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${TensorRT_INCLUDE_Dir}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  src
  .
)

add_library(online SHARED
  src/online.cpp
)

target_link_libraries(online
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBS}
  ${Eigen_LIBRARIES}
  ${PCL_LIBRARIES}
  casadi
  fullyHardNet
  bts
  tensorrtbuffer
  superglue
  calibration_dk
  glob
  patchworkpp
  pclomp
)

add_executable(${PROJECT_NAME} src/online_node.cpp)

target_link_libraries(${PROJECT_NAME}
  online
)