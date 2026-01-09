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

打开Microsoft Store，商店内直接搜索Ubuntu, 注意千万别装Ubuntu20.04

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

```
错误地将C:\Users\liang\AppData\Local\Packages\ 删除解决办法：
https://blog.csdn.net/m0_69593912/article/details/143580502
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

修改xrdp端口为3390
    sudo vim /etc/xrdp/xrdp.ini

sudo service xrdp restart              # 重启服务 

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


从Windows连接图形界面：目前没啥用
    打开Windows “远程桌面连接” 应用
    输入地址：localhost:3390
    登录WSL的用户名和密码
    成功后将进入Ubuntu的Xfce4桌面环境
    远程连接可看到桌面

```


- 安装isaacgym之前
```
sudo apt install plocate # locate xxx命令用于找到系统上的内容， 安装比较费时，估计本地需要建立某种索引。
在Windows上输入WSL Settings 将WSL网络模式从Nat修改为Mirrored https://zhuanlan.zhihu.com/p/15762609815
sudo apt-get install g++ gcc make
libtinfo5的问题 ： sudo ln -s /usr/lib/x86_64-linux-gnu/libtinfo.so.6 /usr/lib/x86_64-linux-gnu/libtinfo.so.5


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


## 配置CooHOI

- Windows上打开VSCode,在Extensions中搜索Remove-WSL出来WSL，安装。
VSCode可直接操作linux目录了

```
New Terminal -> wsl
cd ~

code . # 会安装vscode wsl server

在Extensions中搜索Python
点击Python Debugger, 在右侧中点击 Install in wsl


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

添加国内源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/



学习isaacgym api https://docs.robotsfan.com/isaacgym/programming/simsetup.html
学习rl games api https://github.com/Denys88/rl_games/tree/master/docs

cd ~/Motion
mv isaacgym CooHOI/isaacgym


创建环境coohoi
conda create -n coohoi python=3.8

conda activate coohoi # 如果这一步提示先要执行conda init，可先执行code .配置visual studio code，安装Python Debugger extension, 再Ctrl+Shift+P选择新建的conda env对应的python，再新建一个launch.json来调试python。

安装isaacgym
拷贝isaacgym目录到桌面/Motion/CooHOI/中。
pip install -e isaacgym/python --use-pep517 # 会安装cuda, torch等

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
                "LD_LIBRARY_PATH": "/usr/lib/wsl/lib:\"${env:CONDA_PREFIX}/lib\":\"${env:LD_LIBRARY_PATH}\""
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


再执行 报错
    internal error : libcuda.so!
    [Warning] [carb.gym.plugin] Failed to create a PhysX CUDA Context Manager. Falling back to CPU.
解决办法：which libcuda.so将找到的路径比如/usr/lib/wsl/lib添加到环境变量LD_LIBRARY_PATH中。 环境变量可以添加到launch.json的"env"中。
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "LD_LIBRARY_PATH": "/usr/lib/wsl/lib:\"${env:CONDA_PREFIX}/lib\":\"${env:LD_LIBRARY_PATH}\""
            },



https://forums.developer.nvidia.com/t/wsl2-and-isaac-gym-problem/192069/14

```

## 导出CooHOI模型为onnx
```
pip install onnxruntime
pip install onnx
    # 这个会安装更新的protobuf版本，导致coohoi启动报错。
    解决办法：回退protobuf的版本 pip install protobuf==3.20.0
    


新建amp_network_builder_onnx.py文件，添加如下内容
    from torch import nn
    class AMPBuilderONNX(nn.Module):
        def __init__(self, input_model):
            super().__init__()
            self.model = input_model
            return
    
        def forward(self, obs):
            actor_outputs = self.model.eval_actor(obs)
            value = self.model.eval_critic(obs)
            output = actor_outputs + (value,)
            return output


在common_player.py的run函数开头添加
    # code to write out onnx
    device = 'cpu'
    obs = torch.zeros((1, 299,), dtype=torch.float32, device=device)
    tensor_model = AMPBuilderONNX(self.model.a2c_network)
    tensor_model.train(False)
    tensor_model.to(device)
    torch.onnx.export(tensor_model,(obs,),f"/home/liangjin/Desktop/Motion/coohoi_twoagents.onnx", input_names=("obs",), output_names=("mu","sigma","value"))


```
