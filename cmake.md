# [VSCode + CMake + C++]
- Windows上用gcc编译
- [知乎链接](https://zhuanlan.zhihu.com/p/45528705)
- [github模版工程](https://github.com/1079805974/CppProjectTemplate)
- 配置C++相关
    - c_cpp_properties.json
    ```
    {
        "configurations": [
            {
                "name": "ChessE",
                "includePath": [
                    "${workspaceFolder}/include",
                    "/usr/local/Cellar/cppunit/1.14.0/include/"
                ],
                "defines": [
                    "_DEBUG",
                    "UNICODE",
                    "_UNICODE"
                ],
                "intelliSenseMode": "gcc-x64"       
            }
        ],
        "version": 4
    }
    ```
    - launch.json
    - settings.json
    ```
    {
        "cmake.configureOnOpen": true,
        "cmake.debugConfig": {
            "MIMode": "gdb"
        },
        "C_Cpp.configurationWarnings": "Disabled",
        "C_Cpp.intelliSenseEngineFallback": "Disabled",
        "C_Cpp.errorSquiggles": "Disabled"
    }
    ```
    - tasks.json
    
- 如何编译
    - 配置c_cpp_properties.json
        - name
        - includePath
        - defines
        - intelliSenseMode
    - 方式1
        装CMake插件，然后通过界面上的tab及按钮触发
    - 方式2
        命令行方式
        cd build
        cmake ..
        make
- 如何调试
    - 添加一个可执行文件项目需要在tasks.json中进行配置
        - 

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


        
