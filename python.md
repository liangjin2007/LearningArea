## vscode hotkey
Ctrl + Shift + X : open extention window
Ctrl + Shift + D : open Run And Debug window, gear icon可设置launch.json
Ctrl + Shift+ P: 可设置python interpreter

## directly debug python code in

安装vscode 


https://blog.csdn.net/weixin_44064908/article/details/128393941
https://blog.csdn.net/qq_39299612/article/details/132040787

launch.json example
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "pythonPath": "E:/Anaconda3/envs/env_name/python.exe",
            "args": [
                "--conf", "value1",
                "--path", "value2"
            ],
            "justMyCode": false,
            "stopOnEntry": true
        }
    ]
}

```


