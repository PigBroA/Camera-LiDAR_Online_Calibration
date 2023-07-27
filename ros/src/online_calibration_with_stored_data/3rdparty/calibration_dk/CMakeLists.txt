cmake_minimum_required(VERSION 3.6)
project(calibration_dk)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(calibration_dk SHARED src/parseBin.cpp src/util.cpp)

target_link_libraries(calibration_dk ${OpenCV_LIBS})