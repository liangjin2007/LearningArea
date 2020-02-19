# 工程

- 在线流程图/思维导图制作[processon](https://www.processon.com/)
- 看板[Trello](https://trello.com/b/bwqk2uTp/jinl-roadmap)


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

请下载参考资料中的模版工程进行更改，不能保证一定成功。此处忽略。

- Mac
    - 安装单元测试库cppunit
        - brew install cppunit
    - 从[github](https://github.com/liangjin2007/vscode_cpp_template)上下载代码到本地
    - 使用vscode打开

    - 调试chess
        - 从vscode左侧tab打开调试页面，选择chess，点击启动按钮，程序会自动断点到main函数第一行。
        - 从调试菜单点击第一个或者第二个菜单项。

    - 关闭断到main函数第一行
        - launch.json中设置stopAtEntry为false

    - 关闭调试
        - 在CMakeLists.txt中设置ET(CMAKE_BUILD_TYPE "Release")

    - 日志输出
        - 会新起独立的terminal，显示print结果。
