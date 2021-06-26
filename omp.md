# omp 

它其实是一个CPU上的多线程加速库。通过预编译指令来控制。https://zhuanlan.zhihu.com/p/51173703

https://blog.csdn.net/u011808673/article/details/80319792

- CMake中添加对openmp的支持
```
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
```
