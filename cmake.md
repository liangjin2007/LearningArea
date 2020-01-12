# CMake

## Functions

### projects
- cmake_minimum_required(VERSION 3.7)

- message(STATUS xxx)

- project()

- find_package(OpenCV REQUIRED)

- set(a b)
- unset()

- source_group
Defining a grouping of sources in IDE.

- set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

- SET(CMAKE_C_COMPILER "/usr/bin/gcc")
- SET(CMAKE_CXX_COMPILER "/usr/bin/g++")
- SET(CMAKE_BUILD_TYPE "Debug")                     
- SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
- SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
- set(CMAKE_VERBOSE_MAKEFILE ON)
- SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

- include_directories()
- link_directories()
- add_definitions(-DDEBUG)
- add_definitions(-Wwritable-strings)
- add_executable(exe_name hello.cpp)
- add_library(library_name hello.cpp)
- target_link_libraries(libarry_name dependencies)
- add_subdirectory()
- AUX_SOURCE_DIRECTORY(src DIR_SRCS)

- add_custom_command(TARGET xxxx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/jni/xxx.java
    ${CMAKE_CURRENT_BINARY_DIR}/xxx.java)
- enable_testing()

- install
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfig.cmake" DESTINATION "share/colmap")

- configure_file

- add_custom_target

- set_target_properties

- set_property

- Cache变量
set(Ceres_INCLUDE_DIRS ${CURRENT_ROOT_INSTALL_DIR}/include CACHE PATH "Ceres include direcctories" FORCE)


### utils
- string
e.g. string(REPLACE "/" "\\" GROUP_NAME ${SRC_DIR})

- file
    - file(READ
    - file(WRITE | APPEND
    - file(GLOB
    - file(TOUCH
    - file(STRINGS

- list
    - list(APPEND A B)
    

### macro
e.g.
```
macro(COLMAP_ADD_SOURCE_DIR SRC_DIR SRC_VAR)
    # Create the list of expressions to be used in the search.
    set(GLOB_EXPRESSIONS "")
    foreach(ARG ${ARGN})
        list(APPEND GLOB_EXPRESSIONS ${SRC_DIR}/${ARG})
    endforeach()
    # Perform the search for the source files.
    file(GLOB ${SRC_VAR} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
         ${GLOB_EXPRESSIONS})
    # Create the source group.
    string(REPLACE "/" "\\" GROUP_NAME ${SRC_DIR})
    source_group(${GROUP_NAME} FILES ${${SRC_VAR}})
    # Clean-up.
    unset(GLOB_EXPRESSIONS)
    unset(ARG)
    unset(GROUP_NAME)
endmacro(COLMAP_ADD_SOURCE_DIR)
```


## VARIABLES
- PROJECT_SOURCE_DIR

- CMAKE_CURRENT_SOURCE_DIR

## CONTROLS
- if
e.g.
    - if(a STREQUAL b)
    - if(a MATCHES b)
    - if(a AND NOT b)
    
- foreach 
foreach(SOURCE_FILE ${ARGN})

endforeach()




