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



        
