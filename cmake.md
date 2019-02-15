# cmake
cmake_minimum_required(VERSION 3.7)

message()

project()

find_package(OpenCV REQUIRED)

include_directories()
link_directories()
add_executable(exe_name hello.cpp)
add_library(library_name hello.cpp)
target_link_libraries(libarry_name dependencies)

add_subdirectory

add_custom_command(TARGET xxxx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/jni/xxx.java
    ${CMAKE_CURRENT_BINARY_DIR}/xxx.java)

# [VSCode + CMake + C++](https://zhuanlan.zhihu.com/p/45528705)
