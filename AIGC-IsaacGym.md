## 目标跑起来 CooHOI 
- https://github.com/Winston-Gu/CooHOI?tab=readme-ov-file

- NeuralSync
https://github.com/AnimaVR/NeuroSync_Player?tab=readme-ov-file


### Windows 11安装Ubuntu linux操作系统
- https://blog.csdn.net/bule_shake/article/details/135992375
```
在搜索栏搜索Windows，点击“启动或关闭Windows功能”。

勾上适用于windows的linux子系统

完成后需要重启电脑

打开Microsoft Store，商店内直接搜索Ubuntu

选择第一个下载安装

下载完成后，在桌面搜索栏搜索“Ubuntu”并打开

会报错。

以管理员身份打开PowerShell或CMD，依次执行：
    wsl --update
    重启系统
    
    # 启用适用于Linux的Windows子系统
    dism.exe  /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

    # 启用虚拟机平台

    dism.exe  /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

    # 设置WSL2为默认版本
    wsl --set-default-version 2

在桌面搜索栏搜Ubuntu，此次提示正在安装，过一会会提示设置UNIX username和密码，成功。
    
```

- 安装图形化界面及支持远程桌面连接
```
安装必要组件（在WSL终端中执行）：

sudo apt update && sudo apt upgrade -y
sudo apt install xorg xfce4 xrdp -y
xorg：基础显示服务
xfce4：轻量级桌面环境
xrdp：远程桌面协议服务
配置Xrdp服务：

sudo echo xfce4-session > ~/.xsession  # 设置默认会话为Xfce4
sudo service xrdp restart              # 重启服务 

修改xrdp端口为3390
    sudo vim /etc/xrdp/xrdp.ini  

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

从Windows连接图形界面：

打开Windows “远程桌面连接” 应用
输入地址：localhost:3390
登录WSL的用户名和密码
成功后将进入Ubuntu的Xfce4桌面环境


远程连接可看到桌面

要在wsl2上跑isaacgym，还不够
https://www.cnblogs.com/erbws/p/18888083#fn1

安装cuda https://developer.nvidia.com/cuda-12-1-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb
    sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb
    sudo cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda



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
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
    https://developer.nvidia.com/cuda-12-1-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local

tar -xvzf isaac-gym-preview-4

安装Anaconda3
chmod +x Anaconda3-2025.06-0-Linux-x86_64.sh
./Anaconda3-2025.06-0-Linux-x86_64.sh


学习isaacgym api https://docs.robotsfan.com/isaacgym/programming/simsetup.html
学习rl games api https://github.com/Denys88/rl_games/tree/master/docs

cd ~/Motion
mv isaacgym CooHOI/isaacgym


创建环境coohoi
conda create -n coohoi python=3.8
conda activate coohoi

安装isaacgym
拷贝isaacgym目录到桌面/Motion/CooHOI/中。
pip install -e isaacgym/python --use-pep517

安装其他依赖
pip install -r requirements.txt 
提示找不到torch版本1.8.1, 注释掉requirements.txt中第一行 #torch==1.8.1，因为装isaacgym时已经装了pytorch
提示安装成功。
    
sudo sh cuda_12.1.1_530.30.02_linux.run




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
                "--test",
                "--task","HumanoidAMPCarryObject",
                "--num_envs","16",
                "--cfg_env","/home/liangjin/Desktop/Motion/CooHOI/coohoi/data/cfg/humanoid_carrybox.yaml",
                "--cfg_train","/home/liangjin/Desktop/Motion/CooHOI/coohoi/data/cfg/train/amp_humanoid_task.yaml",
                "--motion_file","/home/liangjin/Desktop/Motion/CooHOI/coohoi/data/motions/coohoi_data/coohoi_data.yaml",
                "--checkpoint","/home/liangjin/Desktop/Motion/CooHOI/coohoi/data/models/SingleAgent.pth"
            ],
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "LD_LIBRARY_PATH": "\"${env:CONDA_PREFIX}/lib\":\"${env:LD_LIBRARY_PATH}\""
            },
            "justMyCode":false,
            "stopOnEntry": true
        }
    ]
}

VSCode中找到coohoi/run.py使它为当前active file
F5调试

报错ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
解决办法：
    sudo cp /home/liangjin/anaconda3/envs/coohoi/lib/libpython3.8.so.1.0 /usr/lib/x86_64-linux-gnu
    
执行 再报错 GLIbCXX_xxx找不到

cd /home/liangjin/anaconda3/envs/coohoi/lib
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6

再执行 报错
run.py: error: unrecognized arguments:
    launch.json中"--test"后不要接空的""
    /Users/liangjin改成/home/liangjin


再执行 报错
    internal error : libcuda.so!
    [Warning] [carb.gym.plugin] Failed to create a PhysX CUDA Context Manager. Falling back to CPU.
    Physics Engine: PhysX
    Physics Device: cpu
    GPU Pipeline: disabled
解决办法：安装vulkan
    wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
    sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.4.313-noble.list https://packages.lunarg.com/vulkan/1.4.313/lunarg-vulkan-1.4.313-noble.list
    sudo apt update
    sudo apt install vulkan-sdk
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.x86_64.json


```



