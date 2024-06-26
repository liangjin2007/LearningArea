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

- 如何用c扩展Python
https://docs.python.org/zh-cn/3/extending/extending.html




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
- CMake教程简书 https://www.jianshu.com/p/3078a4a195df

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
- remove_definitions(-DDEBUG)
- target_compile_definitions https://blog.csdn.net/qq_34369618/article/details/96358204
例子
```
target_compile_definitions(mylib 
    PRIVATE -DMYDLL_EXPORT 
    PUBLIC -DDEBUG)
```
- target_compile_options 
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

- 设置工作目录 set_target_properties(OriginalTressFXSample PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_HOME_DIRECTORY}/bin")

- install
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfig.cmake" DESTINATION "share/colmap")

- configure_file

- add_custom_target

- set_target_properties

- set_property

- Cache变量
set(Ceres_INCLUDE_DIRS ${CURRENT_ROOT_INSTALL_DIR}/include CACHE PATH "Ceres include direcctories" FORCE)


- CPack打包成msi
  - 安裝Wix
  - 需要先写好install部分
  ```
  set(AppName xxx)
  install(DIRECTORY 
    <src dir> 
    DESTINATION <dest relative dir>
    COMPONENT ${AppName})
  install(FILES 
    <src file>
    DESTINATION <dest relative dir>
    COMPONENT ${AppName})
  ```
  - 然后写代码
  ```
  # Cpack
  include (InstallRequiredSystemLibraries)
  set(CPACK_GENERATOR WIX)
  set(CPACK_PACKAGE_NAME xxx)
  set(CPACK_PACKAGE_VERSION_MAJOR 0.1)
  set(CPACK_PACKAGE_VENDOR "xxx.com")
  set(CPACK_PACKAGE_DIRECTORY "build")
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "xxx")
  set(CPACK_PACKAGE_EXECUTABLES "xxx" "xxx" ${CPACK_PACKAGE_EXECUTABLES})
  set(CPACK_CREATE_DESKTOP_LINKS "xxx" ${CPACK_CREATE_DESKTOP_LINKS})
  include(CPack)
  ```

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

# Latex公式编辑
- 在线公式编辑器 https://www.codecogs.com/latex/eqneditor.php
### 常用Latex语法
#### 上下标
```
\begin{aligned}
\dot{x} & = \sigma(y-x) \\
\dot{y} & = \rho x - y - xz \\
\dot{z} & = -\beta z + xy
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\dot{x}&space;&&space;=&space;\sigma(y-x)&space;\\&space;\dot{y}&space;&&space;=&space;\rho&space;x&space;-&space;y&space;-&space;xz&space;\\&space;\dot{z}&space;&&space;=&space;-\beta&space;z&space;&plus;&space;xy&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\dot{x}&space;&&space;=&space;\sigma(y-x)&space;\\&space;\dot{y}&space;&&space;=&space;\rho&space;x&space;-&space;y&space;-&space;xz&space;\\&space;\dot{z}&space;&&space;=&space;-\beta&space;z&space;&plus;&space;xy&space;\end{aligned}" title="\begin{aligned} \dot{x} & = \sigma(y-x) \\ \dot{y} & = \rho x - y - xz \\ \dot{z} & = -\beta z + xy \end{aligned}" /></a>

#### 求和和积分
```
\begin{aligned}
\sum_{i=1}^{n}x_{i}=\int_{0}^{1}f(x)\mathrm{d}x
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\sum_{i=1}^{n}x_{i}=\int_{0}^{1}f(x)\mathrm{d}x&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\sum_{i=1}^{n}x_{i}=\int_{0}^{1}f(x)\mathrm{d}x&space;\end{aligned}" title="\begin{aligned} \sum_{i=1}^{n}x_{i}=\int_{0}^{1}f(x)\mathrm{d}x \end{aligned}" /></a>

```
\begin{aligned}
\sum_{ 1\leqslant i\leq n \atop 1\leqslant j\leq n }a_{ij}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\sum_{&space;1\leqslant&space;i\leq&space;n&space;\atop&space;1\leqslant&space;j\leq&space;n&space;}a_{ij}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\sum_{&space;1\leqslant&space;i\leq&space;n&space;\atop&space;1\leqslant&space;j\leq&space;n&space;}a_{ij}&space;\end{aligned}" title="\begin{aligned} \sum_{ 1\leqslant i\leq n \atop 1\leqslant j\leq n }a_{ij} \end{aligned}" /></a>

#### 极限
```
\begin{aligned}
\lim_{n \to \infty }\sin x_{n}=0
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\lim_{n&space;\to&space;\infty&space;}\sin&space;x_{n}=0&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\lim_{n&space;\to&space;\infty&space;}\sin&space;x_{n}=0&space;\end{aligned}" title="\begin{aligned} \lim_{n \to \infty }\sin x_{n}=0 \end{aligned}" /></a>

#### 根式
```
\begin{aligned}
x=\sqrt[m]{1+x^{p}}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;x=\sqrt[m]{1&plus;x^{p}}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;x=\sqrt[m]{1&plus;x^{p}}&space;\end{aligned}" title="\begin{aligned} x=\sqrt[m]{1+x^{p}} \end{aligned}" /></a>

#### 绝对值
```
\begin{aligned}
\left | a+b \right |=\coprod_{m}^{n} \frac{e}{f}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\left&space;|&space;a&plus;b&space;\right&space;|=\coprod_{m}^{n}&space;\frac{e}{f}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\left&space;|&space;a&plus;b&space;\right&space;|=\coprod_{m}^{n}&space;\frac{e}{f}&space;\end{aligned}" title="\begin{aligned} \left | a+b \right |=\coprod_{m}^{n} \frac{e}{f} \end{aligned}" /></a>

#### 矩阵
```
\begin{aligned}
\begin{pmatrix} 1 & 3 & 5 \\
2 & 4 & 6
\end{pmatrix}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\begin{pmatrix}&space;1&space;&&space;3&space;&&space;5&space;\\&space;2&space;&&space;4&space;&&space;6&space;\end{pmatrix}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\begin{pmatrix}&space;1&space;&&space;3&space;&&space;5&space;\\&space;2&space;&&space;4&space;&&space;6&space;\end{pmatrix}&space;\end{aligned}" title="\begin{aligned} \begin{pmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{pmatrix} \end{aligned}" /></a>

#### 花括号
```
\begin{aligned}
\overbrace{a+b+\cdots +y+z}^{26}_{=\alpha +\beta}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\overbrace{a&plus;b&plus;\cdots&space;&plus;y&plus;z}^{26}_{=\alpha&space;&plus;\beta}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\overbrace{a&plus;b&plus;\cdots&space;&plus;y&plus;z}^{26}_{=\alpha&space;&plus;\beta}&space;\end{aligned}" title="\begin{aligned} \overbrace{a+b+\cdots +y+z}^{26}_{=\alpha +\beta} \end{aligned}" /></a>

```
\begin{aligned}
a+\underbrace{b+\cdots +y}_{24}+z 
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;a&plus;\underbrace{b&plus;\cdots&space;&plus;y}_{24}&plus;z&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;a&plus;\underbrace{b&plus;\cdots&space;&plus;y}_{24}&plus;z&space;\end{aligned}" title="\begin{aligned} a+\underbrace{b+\cdots +y}_{24}+z \end{aligned}" /></a>

#### 堆砌
```
\begin{aligned}
y\stackrel{\rm def}{=} f(x) \stackrel{x\rightarrow 0}{\rightarrow} A
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;y\stackrel{\rm&space;def}{=}&space;f(x)&space;\stackrel{x\rightarrow&space;0}{\rightarrow}&space;A&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;y\stackrel{\rm&space;def}{=}&space;f(x)&space;\stackrel{x\rightarrow&space;0}{\rightarrow}&space;A&space;\end{aligned}" title="\begin{aligned} y\stackrel{\rm def}{=} f(x) \stackrel{x\rightarrow 0}{\rightarrow} A \end{aligned}" /></a>

#### 乘/除/点击
```
\begin{aligned}
a \cdot b \\
a  \times b \\
a   \div   b
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;a&space;\cdot&space;b&space;\\&space;a&space;\times&space;b&space;\\&space;a&space;\div&space;b&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;a&space;\cdot&space;b&space;\\&space;a&space;\times&space;b&space;\\&space;a&space;\div&space;b&space;\end{aligned}" title="\begin{aligned} a \cdot b \\ a \times b \\ a \div b \end{aligned}" /></a>

#### 连乘/连加
```
\begin{aligned}
\prod  _{a}^{b} \\
\prod \limits_{i=1}^{n} \\
\sum _{a}^{b} \\
\sum \limits_{i=1} ^{n}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\prod&space;_{a}^{b}&space;\\&space;\prod&space;\limits_{i=1}^{n}&space;\\&space;\sum&space;_{a}^{b}&space;\\&space;\sum&space;\limits_{i=1}&space;^{n}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\prod&space;_{a}^{b}&space;\\&space;\prod&space;\limits_{i=1}^{n}&space;\\&space;\sum&space;_{a}^{b}&space;\\&space;\sum&space;\limits_{i=1}&space;^{n}&space;\end{aligned}" title="\begin{aligned} \prod _{a}^{b} \\ \prod \limits_{i=1}^{n} \\ \sum _{a}^{b} \\ \sum \limits_{i=1} ^{n} \end{aligned}" /></a>

#### 大于等于 小于等于 不等于
```
\begin{aligned}
a \geq b \\
a \leq b \\
a \neq b
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;a&space;\geq&space;b&space;\\&space;a&space;\leq&space;b&space;\\&space;a&space;\neq&space;b&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;a&space;\geq&space;b&space;\\&space;a&space;\leq&space;b&space;\\&space;a&space;\neq&space;b&space;\end{aligned}" title="\begin{aligned} a \geq b \\ a \leq b \\ a \neq b \end{aligned}" /></a>

#### 积分，正负无穷
```
\begin{aligned}
\int_a^b \\
\int_{- \infty}^{+ \infty}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\int_a^b&space;\\&space;\int_{-&space;\infty}^{&plus;&space;\infty}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\int_a^b&space;\\&space;\int_{-&space;\infty}^{&plus;&space;\infty}&space;\end{aligned}" title="\begin{aligned} \int_a^b \\ \int_{- \infty}^{+ \infty} \end{aligned}" /></a>

#### 子集
```
\begin{aligned}
A \subset B \\
A \not \subset B \\
A \subseteq B \\
A \subsetneq B \\
A \subseteqq B \\
A \subsetneqq B \\
A \supset B
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;A&space;\subset&space;B&space;\\&space;A&space;\not&space;\subset&space;B&space;\\&space;A&space;\subseteq&space;B&space;\\&space;A&space;\subsetneq&space;B&space;\\&space;A&space;\subseteqq&space;B&space;\\&space;A&space;\subsetneqq&space;B&space;\\&space;A&space;\supset&space;B&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;A&space;\subset&space;B&space;\\&space;A&space;\not&space;\subset&space;B&space;\\&space;A&space;\subseteq&space;B&space;\\&space;A&space;\subsetneq&space;B&space;\\&space;A&space;\subseteqq&space;B&space;\\&space;A&space;\subsetneqq&space;B&space;\\&space;A&space;\supset&space;B&space;\end{aligned}" title="\begin{aligned} A \subset B \\ A \not \subset B \\ A \subseteq B \\ A \subsetneq B \\ A \subseteqq B \\ A \subsetneqq B \\ A \supset B \end{aligned}" /></a>

#### 上滑线，下滑线
```
\begin{aligned}
\overline {a+b} \\
\underline {a+b}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\overline&space;{a&plus;b}&space;\\&space;\underline&space;{a&plus;b}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\overline&space;{a&plus;b}&space;\\&space;\underline&space;{a&plus;b}&space;\end{aligned}" title="\begin{aligned} \overline {a+b} \\ \underline {a+b} \end{aligned}" /></a>

#### 矢量
```
\begin{aligned}
\vec {ab} \\
\overrightarrow{ab}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\vec&space;{ab}&space;\\&space;\overrightarrow{ab}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\vec&space;{ab}&space;\\&space;\overrightarrow{ab}&space;\end{aligned}" title="\begin{aligned} \vec {ab} \\ \overrightarrow{ab} \end{aligned}" /></a>
