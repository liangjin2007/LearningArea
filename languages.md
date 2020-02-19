# Python
- generator yield
- 上下文管理器
  - with
  - contextmanager装饰器
  - https://www.cnblogs.com/zhbzz2007/p/6158125.html
- ndarray.sort
  - [ndarray.sort(order='y')](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ndarray.sort.html#numpy.ndarray.sort)
- list.sort
```
a = [("a",0.1),("b",0.05),("c",0.06)]
a.sort(key=lambda x:x[1], reverse=True)
```
- 取numpy数组中某一个axis
a[0,:] # 取第一行
a[:,:,1] # 2维
a[:,:,0:1] # 3维

- 取字符串子串
a="abcc_def"
a[5:] # "def"

- make tuple longer
a = (None,)+(3,5) # (None, 3, 5)

- 使用字典作为某函数的扩展参数
```
options = {
    "activation": "relu",
    "kernel_size": (3, 3),
    "padding": "same"
}
function_A(**options)
```
- method
    - Python 语法
    @property
    @staticmethod

- 数组操作
```
np.round
np.reshape(a, [-1])
c, d = np.meshgrid([1,2,3,4], [4,5,6,7]) #
[[1 2 3 4]
 [1 2 3 4]
 [1 2 3 4]
 [1 2 3 4]]
[[4 4 4 4]
 [5 5 5 5]
 [6 6 6 6]
 [7 7 7 7]]
 e = np.stack([c,d,c,d], axis=0)
 [[1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]
 [4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7]
 [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]
 [4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7]]
```
- datatime
datetime在python中比较常用，主要用来处理时间日期，使用前先倒入datetime模块。下面总结下本人想到的几个常用功能。
```
print datetime.datetime.now()
datetime.timedelta(days=1)
(datetime.datetime.now() - datetime.datetime.utcnow()).total_seconds()

datetime转str格式：
datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
str格式转datetime格式：
datetime.datetime.strptime("2015-07-17 16:58:46","%Y-%m-%d %H:%M:%S")
```

- slicing ::
```
这个是python的slice notation的特殊用法。

a = [0,1,2,3,4,5,6,7,8,9]
b = a[i:j] 表示复制a[i]到a[j-1]，以生成新的list对象
b = a[1:3] 那么，b的内容是 [1,2]
当i缺省时，默认为0，即 a[:3]相当于 a[0:3]
当j缺省时，默认为len(alist), 即a[1:]相当于a[1:10]
当i,j都缺省时，a[:]就相当于完整复制一份a了

b = a[i:j:s]这种格式呢，i,j与上面的一样，但s表示步进，缺省为1.
所以a[i:j:1]相当于a[i:j]
当s<0时，i缺省时，默认为-1. j缺省时，默认为-len(a)-1
所以a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍。所以你看到一个倒序的东东。
```
- numpy数组二维索引取子矩阵
```
>>> a = np.arange(12).reshape(3,4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> i = np.array( [ [0,1],                      # 第一个轴的索引
...                 [1,2] ] )
>>> j = np.array( [ [2,1],                        # 第二个轴的索引
...                 [3,3] ] )
>>>
>>> a[i,j]                                     # i 和 j形状必须相同
array([[ 2,  5],
       [ 7, 11]])
```

# english
## 知乎答案

```
首先你要明白，听懂的过程不是你听懂了。而是你【搜索】到了。这个过程更像是解密的过程。耳朵听到的语言，其实是一段模拟性声音符号。他不是准确的，也不完整。需要你带着听到的信息，在大脑数据库里进行搜索，直到【匹配】。听懂快速英语最大的误区是：追求听清。这是永远无法听清的，因为说话者根本就没有说清楚。（好比北京人说“天安门”，只说“天门”两个字，你永远无法听到“安”字。听懂的原因是：你进行了【语境匹配】。什么是语境匹配？你听到：蓝人爱绿人。第一反应不会是理解成：蓝色的人爱绿色的人。因为这超出你脑子里的数据库范围了，没有这样的话。而匹配度最高的，是：男人爱女人。这就是听懂中国方言的过程。听懂外国话，也必须是这样的过程。需要具备：1）脑子预先存在数据库。2）取样点越多，匹配越快。【数据库】是指你拥有的【词汇量】。【取样点】是指你听到的【清晰度】。
```

##词汇量
===============================================================
elaborate 复杂的，精巧的

excessive cost过度的开销

encompass围绕包围 

untenable 站不住脚的

punctual准时 he is always punctual.

approximate近似

anatomy解剖，部位 

degradation屈辱

consensus一致意见

aggregate聚合

hand-crafted feature

correlate with 有关

be confronted to personal tastes

depart from this line of research

deviate from this line of research偏离

appealing for many people 有吸引力的

adhere to 因为。。。

spontaneous自发的

mitigate 缓和

attractiveness吸引力

outperform 超越

complication困难

a gold mine 金矿

amateur业余爱好者
hobbyists业余爱好者

professional专业的

intrinsic value 内涵值

semantic语义

photographic style annotations照片风格标注

consistent 连贯

shutter speed 快门速度

exposure曝光 

complementary colors互补色

color tone色调

exhaustive彻底的，详尽的

color saturation 颜色饱和度

predictive 预测的

aggregate 集成

be presumed to 假定

discriminate区别

vibrant鲜明的

greenery绿色植物

dominate统治，主导

superior to比。。。更好

neutral illumination 中性光照

diversity多样性

diminishing减少

categorization分类

encapsulate 封装

thematic主题的

subjective主观的

# Matlab
- Matlab与Python的区别
   - 行尾 “;”
   - range for i in 1:3
   - for, if 等尾部不需要 ":"
   - 取元素是()而不是[] img(:,:,3) vs img[:,:,3]
   - for end pair
 - example
   ```
   
   ```

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

# Markdown语法

-Markdown语法 https://www.jianshu.com/p/191d1e21f7ed
```
  标题 #我是标题，##我是标题,...
  加粗 **我是粗体**
  斜体 *我是斜体*
  加粗+斜体 ***我是加粗斜体***
  删除线 ~~我是删除线~~
  引用 >我是引用的内容， >>我也是引用的内容
  分割线 
    ***我是分割线， *****我也是分割线， ---我也是分割线， -----我也是分割线------
  图片![图片下面的文字](图片地址 "图片title")
  [超链接名](超链接地址 "超链接title")
  无序列表 +, -, *
  有序列表 数字加点 1. 2. ...
  列表嵌套 下一级列表之前敲三个空格
  表格 
    表头|表头|表头
    ---|:--:|---:
    内容|内容|内容
  代码 
    单行代码 `代码`
    多行代码 ``` 代码 ```
  流程图
    ```flow
    st=>start: 开始
    op=>operation: My Operation
    cond=>condition: Yes or No?
    e=>end
    st->op->cond
    cond(yes)->e
    cond(no)->op
    &```
```
- 应用

[Editor.md](https://pandao.github.io/editor.md/)
可以用于给表格图片url打标，请自己脑补。
