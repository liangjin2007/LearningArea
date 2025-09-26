## 目标跑起来 CooHOI 
- https://github.com/Winston-Gu/CooHOI?tab=readme-ov-file

- NeuralSync
https://github.com/AnimaVR/NeuroSync_Player?tab=readme-ov-file


```
下载
下载https://developer.nvidia.com/isaac-gym-preview-4
解压得到isaacgym目录

下载CooHOI-main.zip
解压放到桌面/Motion/中

下载Anaconda3 https://www.anaconda.com/download
安装Anaconda3

使用VSCode创建环境
VSCode打开CooHOI-main
打开VSCode的Terminal, 此时可以看到(base)环境

创建环境coohoi
conda create -n coohoi python=3.8
conda activate coohoi

安装isaacgym
拷贝isaacgym目录到桌面/Motion/CooHOI-main/中。
pip install -e isaacgym/python 

安装其他依赖
pip install -r requirements.txt 
提示找不到torch版本1.8.1, 注释掉requirements.txt中第一行 #torch==1.8.1，因为装isaacgym时已经装了pytorch
提示安装成功。

设置VSCode可跑SingleAgent
VSCode安装Extensions Python (否则Command + Shift + P中没有Python: Select Interpreter)
Command + Shift + P选择Python: Select Interpreter, 选择python(coohoi)
Create a launch.json Python Debugger: Current File with Arguments如下：
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [

        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "args": [
                "--test","",
                "--task","HumanoidAMPCarryObject",
                "--num_envs","16",
                "--cfg_env","/Users/liangjin/Desktop/Motion/CooHOI-main/coohoi/data/cfg/humanoid_carrybox.yaml",
                "--cfg_train","/Users/liangjin/Desktop/Motion/CooHOI-main/coohoi/data/cfg/train/amp_humanoid_task.yaml",
                "--motion_file","/Users/liangjin/Desktop/Motion/CooHOI-main/coohoi/data/motions/coohoi_data/coohoi_data.yaml",
                "--checkpoint","/Users/liangjin/Desktop/Motion/CooHOI-main/coohoi/data/models/SingleAgent.pth"
            ],
            "envs": [
                "CUDA_VISIBLE_DEVICES","0"
            ],
            "justMyCode":false,
            "stopOnEntry": true
        }
    ]
}

VSCode中找到coohoi/run.py使它为当前active file
设置断点
按Fn + F5开启调试

```

### Windows 11安装Ubuntu linux操作系统
- https://blog.csdn.net/bule_shake/article/details/135992375




