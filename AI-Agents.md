# 目录
- [1.OpenCode](#1OpenCode)

## 1.OpenCode

- 文档 https://opencode.ai/docs/zh-cn/
- 安装
```
安装方式1：
  安装node.js
  powershell:
    通过node.js安装opencode : npm install -g opencode-ai
    输入opencode命令，会打开opencode的TUI
  TUI：
    在TUI中输入/models 按enter, 选择MiniMax 2.5 Free模型
    打开一个已有的工程：cd D:/ue-simulator

安装方式2：
  安装opencode-desktop-windows-x64.exe

我这边是两种方式都装了，貌似在设置API key的时候最好使用TUI。



```

- 简单开始
```
选择MiniMax M2.5 Free 模型
为项目初始化 OpenCode ： /init
Tab切换Plan和Build模式
撤销修改 /undo
重做命令 /redo

免费的模型有限额。
```

- 配置Coding Plan为智谱Coding Plan
```
参考 https://docs.bigmodel.cn/cn/coding-plan/tool/opencode#2-%E8%BF%90%E8%A1%8C-opencode-auth-login-%E5%B9%B6%E9%80%89%E6%8B%A9-zhipu-ai-coding-plan

打开powershell
opencode auth login

Select provider为Zhipu AI Coding Plan, 按enter

提示输入API key
需要先购买


```

