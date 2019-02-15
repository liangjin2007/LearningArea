# Mac上使用VSCode+CMake开发C++程序

### 参考资料：
- [VSCode + CMake + C++](https://zhuanlan.zhihu.com/p/45528705)
- [github模版工程](https://github.com/1079805974/CppProjectTemplate)
- [简书](https://www.jianshu.com/p/050fa455bc74)

### 第一步 安装VSCode插件
- C/C++
- C/C++ Clang Command Adapter
- C/C++ Compile Run
- CMake Tools
- CMake Tools Helper
- CMake

### 第二步 下载代码并用VSCode开始开发
- Windows

请下载参考资料中的模版工程进行更改即可。

- Mac

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


        
