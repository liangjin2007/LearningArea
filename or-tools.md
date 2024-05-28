## 编译
```
1. 从github clone代码

2. 安装Visual Studio 2022

3. 使用CMake GUI打开源代码，设置build目录，配置为Visual Studio 2022 x64项目。 点击Configure会提示absl找不到，到Build中把BUILD_DEPS勾上，会自动去下载第三方依赖。点击Generate。 打开Visual Studio项目，设置为Release X64(Debug X64编译不过)。

4. 编译。会报错：
   1. 将报错的三个文件的编码用Notepad++修改为使用utf-8-BOM编码。
   2. 将两个libscip和scip项目的编译选项修改一下。
   编译成功。
```
整体上这个工程的编译有点慢。
## 知识点
此项目的C++





