
- https://github.com/Winston-Gu/CooHOI?tab=readme-ov-file
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

```
安装Conda
    wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
    安装Anaconda3
    chmod +x Anaconda3-2025.06-0-Linux-x86_64.sh
    ./Anaconda3-2025.06-0-Linux-x86_64.sh
    
    添加国内源
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

    创建环境coohoi
    conda create -n isaacgym python=3.8  # 注意isaacgym需要3.8版本python
    conda activate isaacgym # 如果这一步提示先要执行conda init，可先执行code .配置visual studio code，安装Python Debugger extension, 再Ctrl+Shift+P选择新建的conda env对应的python，再新建一个launch.json来调试python。


安装isaacgym
    wget https://developer.nvidia.com/isaac-gym-preview-4
    tar -xvzf isaac-gym-preview-4
    拷贝isaacgym到想要部署的python代码中，比如mv isaacgym CooHOI/isaacgym
    pip install -e isaacgym/python --use-pep517 # 会安装cuda, torch等
    sudo cp /home/liangjin/anaconda3/envs/isaacgym/lib/libpython3.8.so.1.0 /usr/lib/x86_64-linux-gnu
    

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
