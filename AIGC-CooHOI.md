- https://github.com/Winston-Gu/CooHOI?tab=readme-ov-file

```
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

