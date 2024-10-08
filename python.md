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

## setup编译torch cpp扩展 
```
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```

## tqdm进度条
https://www.cnblogs.com/hsluoyang/p/17456462.html

## safetensors
https://blog.csdn.net/wzk4869/article/details/130668642
```
类似于ONNX这种通用模型，safetensors文件:
safetensors是一种新的文件格式，旨在安全地存储机器学习模型的权重。
这种格式通常用于存储大型模型，比如GPT-2或GPT-3，以提供额外的安全性和透明度。
它可以包含模型的全部或部分权重，并且通常与特定的模型架构相对应。
safetensors格式可能包含有关模型权重的元数据，例如权重的大小、数据类型和结构信息。
这种格式有助于防止模型权重被篡改，确保模型在传输和存储过程中的完整性。
```

