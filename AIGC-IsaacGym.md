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
```
以管理员身份打开PowerShell或CMD，依次执行：

# 启用适用于Linux的Windows子系统
dism.exe  /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
# 启用虚拟机平台
dism.exe  /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
# 设置WSL2为默认版本
wsl --set-default-version 2
```

- 完成以上会进入Enter new UNIX username, passward等。

- 安装图形化界面及支持远程桌面连接
```
安装必要组件（在WSL终端中执行）：

sudo apt update && sudo apt upgrade -y
sudo apt install xorg xfce4 xrdp -y
xorg：基础显示服务
xfce4：轻量级桌面环境
xrdp：远程桌面协议服务 1
配置Xrdp服务：

sudo echo xfce4-session > ~/.xsession  # 设置默认会话为Xfce4
sudo service xrdp restart              # 重启服务 
确保端口默认为3390（WSL默认映射）
从Windows连接图形界面：

打开Windows “远程桌面连接” 应用
输入地址：localhost:3390
登录WSL的用户名和密码
成功后将进入Ubuntu的Xfce4桌面环境

xrdp默认端口查看方法
xrdp的默认端口信息可通过配置文件直接查看，其默认端口为3389。在WSL环境中，为避免与Windows系统远程桌面服务冲突，通常需修改为其他端口（如3390）12。

查看配置文件获取端口
打开xrdp配置文件
通过以下命令编辑xrdp的主配置文件：
sudo vim /etc/xrdp/xrdp.ini  
查找端口设置
在文件中搜索 port 关键字，默认配置行通常为：
port=3389 
该数值即为xrdp的默认监听端口12。
验证端口占用状态
若需确认当前系统中xrdp实际监听的端口，可使用 netstat 或 ss 命令：

linux桌面黑屏的问题：
    一、修改XRDP启动脚本（推荐方案）
    在Ubuntu服务器上操作：
        sudo vim /etc/xrdp/startwm.sh 
    在文件顶部添加以下内容：
        unset DBUS_SESSION_BUS_ADDRESS
        unset XDG_RUNTIME_DIR 
        . $HOME/.profile
    保存后重启xrdp服务：
        sudo systemctl restart xrdp 
远程连接可看到桌面


```    

- Windows上打开VSCode,在Extensions中搜索Remove-WSL出来WSL，安装。
VSCode可直接操作linux目录了

```
New Terminal -> wsl
cd ~
mkdir Motion
cd Motion

下载
wget https://developer.nvidia.com/isaac-gym-preview-4
wget wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
git clone https://github.com/Winston-Gu/CooHOI.git

tar -xvzf isaac-gym-preview-4

安装Anaconda3
chmod +x Anaconda3-2025.06-0-Linux-x86_64.sh
./Anaconda3-2025.06-0-Linux-x86_64.sh


学习isaacgym api https://docs.robotsfan.com/isaacgym/programming/simsetup.html
学习rl games api https://github.com/Denys88/rl_games/tree/master/docs

cd ~/Motion
mv isaacgym CooHOI/isaacgym






cd CooHOI
pip install -e isaacgym/python
```



