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

## importlib
```
importlib 是 Python 的一个标准库，它提供了用于导入模块的函数。通过使用 importlib，你可以以编程方式导入模块，而不是使用 Python 的内置 import 语句。这在以下场景中特别有用：
在运行时动态导入模块。
想要导入的模块名称只有在程序运行后才能确定。
需要重新导入模块，因为 importlib.reload() 函数可以重新加载已经导入的模块。
以下是 importlib 的一些常用功能：
    importlib.import_module(): 动态导入模块。
    importlib.reload(): 重新加载模块。


importlib.util.spec_from_file_location的方式加载一个文件模块
from importlib.util import spec_from_file_location, module_from_spec
import importlib
# 假设我们有一个名为 'mymodule' 的模块在 '/path/to/mymodule.py'
module_name = 'mymodule'
module_path = '/path/to/mymodule.py'
# 创建模块规范
spec = spec_from_file_location(module_name, module_path)
# 使用 spec 创建模块对象
module = module_from_spec(spec)
spec.loader.exec_module(module)
# 现在可以像使用普通模块一样使用 module
```

## tqdm进度条
```
from tqdm import tqdm
import time
for i in tqdm(range(100)):
    # 假设我们正在进行一些耗时的操作，比如训练深度学习模型
    time.sleep(0.01)

更多例子可以看https://www.cnblogs.com/hsluoyang/p/17456462.html
```

## f-string
```
在Python中，字符串前的 f 或者 F 表示这是一个格式化字符串（formatted string literals），也被称为 f-string。这是Python 3.6版本引入的一种新的字符串格式化方法，提供了一种非常简洁和直观的方式来在字符串中嵌入表达式。
使用 f-string，你可以在字符串中直接包含表达式，表达式会被自动求值，并将结果嵌入到字符串中。下面是一个简单的例子：
    name = "Alice"
    age = 25
    greeting = f"Hello, {name}. You are {age} years old."
    print(greeting)
输出将会是：
    Hello, Alice. You are 25 years old.
在这个例子中，{name} 和 {age} 是占位符，它们会被变量 name 和 age 的值所替换。
f-string还支持更复杂的表达式，例如：
    price = 49.99
    taxed_price = price * 1.08
    formatted_price = f"The price with tax is: ${taxed_price:.2f}"
    print(formatted_price)
输出将会是：
    The price with tax is: $53.99
```

## 

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

