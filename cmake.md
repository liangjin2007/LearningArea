# VSCode上使用CMake开发C++程序


为了在Mac机器上使用现在非常流形的编程环境VSCode来开发C++程序，目前没有可以拿来即用的开源代码或者比较详细的文档。这里通过网络上收集资料及自己整合，开发了一个在Mac上能直接下载下来就能使用的模版。

### 参考资料：
- [VSCode + CMake + C++](https://zhuanlan.zhihu.com/p/45528705)
- [github模版工程](https://github.com/1079805974/CppProjectTemplate)
- [简书](https://www.jianshu.com/p/050fa455bc74)

### Windows上开发
请下载参考资料中的

### 第一步 安装VSCode插件
- C/C++
- C/C++ Clang Command Adapter
- C/C++ Compile Run
- CMake Tools
- CMake Tools Helper
- CMake

### 第二步 下载代码并用VSCode开始开发
从[github](https://github.com/1079805974/CppProjectTemplate)上下载代码到本地。此代码与参考资料中的代码的区别是我修改了launch.json
```
{
    "name": "chess",
    "type": "cppdbg",
    "request": "launch",
    "program": "${workspaceFolder}/bin/chess",
    "args": [],
    "preLaunchTask": "build",
    "stopAtEntry": true,
    "cwd": "${workspaceFolder}",
    "environment": [],
    "externalConsole": true,
    "MIMode": "lldb"
}
```


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


        
