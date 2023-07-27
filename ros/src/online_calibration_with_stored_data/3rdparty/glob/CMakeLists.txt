cmake_minimum_required(VERSION 3.6)
project(glob)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(glob SHARED src/glob.cpp)