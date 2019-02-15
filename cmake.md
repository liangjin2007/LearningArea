# CMake
cmake_minimum_required(VERSION 3.7)

message()

project()

find_package(OpenCV REQUIRED)

PROJECT_SOURCE_DIR

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_C_COMPILER "/usr/bin/gcc")
SET(CMAKE_CXX_COMPILER "/usr/bin/g++")
SET(CMAKE_BUILD_TYPE "Debug")                     
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_VERBOSE_MAKEFILE ON)
SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

include_directories()
link_directories()
add_definitions(-DDEBUG)
add_definitions(-Wwritable-strings)
add_executable(exe_name hello.cpp)
add_library(library_name hello.cpp)
target_link_libraries(libarry_name dependencies)

AUX_SOURCE_DIRECTORY(src DIR_SRCS)
add_executable(chess ${DIR_SRCS})

add_subdirectory()

add_custom_command(TARGET xxxx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/jni/xxx.java
    ${CMAKE_CURRENT_BINARY_DIR}/xxx.java)

enable_testing()
