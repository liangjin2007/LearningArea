- 源代码地址 https://github.com/xbpeng/MimicKit?tab=readme-ov-file
- 部署
```
安装isaacgym
安装requirements.txt
下载data放到MimicKit/data/中。这一步在wsl中，我是直接从Windows下载，并拷贝到WSL Ubuntu的MimicKit/data/中。
```
- 使用WSL训练（因为isaacgym只能linux系统）
- VSCode launch.json
```
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
                "--mode", "train",
                "--num_envs", "1",
                "--env_config", "/home/liangjin/Desktop/MimicKit-main/data/envs/deepmimic_humanoid_env.yaml",
                "--agent_config", "/home/liangjin/Desktop/MimicKit-main/data/agents/deepmimic_humanoid_ppo_agent.yaml",
                "--visualize", "false",
                "--log_file", "output/model.pt"
            ],
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "LD_LIBRARY_PATH": "/usr/lib/wsl/lib:\"${env:CONDA_PREFIX}/lib\":\"${env:LD_LIBRARY_PATH}\""
            },
        }
    ]
}
```

- 转pkl为json
```
import os
import pickle
import json
import numpy as np

in_pickle_file = '/home/liangjin/Desktop/MimicKit/data/motions/smpl/smpl_walk.pkl'
out_json_file = '/home/liangjin/Desktop/smpl_walk.json'

with open(in_pickle_file, "rb") as filestream:
    in_dict = pickle.load(filestream)

    loop_mode_val = in_dict["loop_mode"]
    fps = in_dict["fps"]
    frames = in_dict["frames"]
    frames = np.array(frames, dtype=np.float32)

    json_dict = {}

    trans = frames[..., 0:3]
    pose_aa = frames[..., 3:]

    json_dict['trans'] = trans.tolist()
    json_dict['pose_aa'] = pose_aa.tolist()

    json_str = json.dumps(json_dict)
    json_binary = json_str.encode("utf-8")
    with open(out_json_file,  'wb') as f:
        f.write(json_binary)     
```




### 支持newton物理引擎
```
要安装newton, 需要python=3.10以上。与之前安装的isaacgym需要python=3.8冲突。

# 创建新conda环境
conda create -n mimickit_newton python==3.10
conda activate mimickit_newton.

# 安装newton
git clone https://github.com/newton-physics/newton
cd newton
git checkout cde9610aff71995d793f9b60e6dc26299e29885c
（官网推荐使用uv进行安装。)等于我还得再装个uv来管理环境，导致跟已经装了的conda会产生冲突。

# 安装newton其他依赖
pip install mujoco --pre -f https://py.mujoco.org/
pip install warp-lang --pre -U -f https://pypi.nvidia.com/warp-lang/
pip install git+https://github.com/google-deepmind/mujoco_warp.git@main
pip install pyglet

尝试直接pip install -e . 来安装newton，也提示newton-0.2.0安装成功。

# 更新mimickit代码
git pull

# 修改vscode中的launch.json，添加新的参数
--engine_config","/home/liangjin/Desktop/MimicKit/data/engines/isaac_gym_engine.yaml"
另外，为了调试set_root_pos等接口实际是怎么设置到mujoco中的，添加
"--devices", "cpu"

vscode launch.json内容
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
            // train parameters for training humanoid which has 15 joints in isaacgym engine
            // "args": [
            //     "--mode", "train",
            //     "--num_envs", "1",
            //     "--env_config", "/home/liangjin/Desktop/MimicKit/data/envs/deepmimic_humanoid_env.yaml",
            //     "--agent_config", "/home/liangjin/Desktop/MimicKit/data/agents/deepmimic_humanoid_ppo_agent.yaml",
            //     "--visualize", "true",
            //     "--engine_config","/home/liangjin/Desktop/MimicKit/data/engines/isaac_gym_engine.yaml",
            //     "--log_file", "output/model.pt"
            // ],
            //
            // train parameters for training smpl which has 24 joints in isaacgym engine.
            // "args": [
            //     "--mode", "train",
            //     "--num_envs", "1",
            //     "--env_config", "/home/liangjin/Desktop/MimicKit/data/envs/deepmimic_smpl_env.yaml",
            //     "--agent_config", "/home/liangjin/Desktop/MimicKit/data/agents/deepmimic_smpl_ppo_agent.yaml",
            //     "--visualize", "true",
            //     "--engine_config","/home/liangjin/Desktop/MimicKit/data/engines/isaac_gym_engine.yaml",
            //     "--log_file", "output/model.pt"
            // ],
            //
            // train parameters for training smpl which has 24 joints in newton engine.
            "args": [
                "--mode", "train",
                "--num_envs", "1",
                "--env_config", "/home/liangjin/Desktop/MimicKit/data/envs/deepmimic_smpl_env.yaml",
                "--agent_config", "/home/liangjin/Desktop/MimicKit/data/agents/deepmimic_smpl_ppo_agent.yaml",
                "--visualize", "true",
                "--engine_config","/home/liangjin/Desktop/MimicKit/data/engines/newton_engine.yaml",
                "--log_file", "output/model.pt",
                "--devices", "cpu"
            ],
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "LD_LIBRARY_PATH": "/usr/lib/wsl/lib:\"${env:CONDA_PREFIX}/lib\":\"${env:LD_LIBRARY_PATH}\""
            },
        }
    ]
}


# VSCode需要使用Ctrl+Shift+P重新选择Python Interpretor为新的mimickit_newton环境下的Python

# 选择mimickit/run.py 按F5启动训练

```




- 报错
```
碰到以下代码报错:
    gt = torch.utils.cpp_extension.load(name="gymtorch", sources=sources, extra_cflags=cflags, verbose=True)

    尝试安装gcc，没解决问题:
        # 安装GCC 10
        sudo apt update
        sudo apt install gcc-10 g++-10 -y
        
        # 设置为默认编译器
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100
        
        # 验证版本
        gcc --version  # 应显示gcc (Ubuntu 10.3.0-1ubuntu1~20.04) 10.3.0
    
    安装编译需要的东西：
        sudo apt update && sudo apt install -y build-essential
        c++ --version成功
        which c++ 返回/usr/bin/c++

libcuda.so的问题：
    launch.json添加
    "env": {
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/usr/lib/wsl/lib:\"${env:CONDA_PREFIX}/lib\":\"${env:LD_LIBRARY_PATH}\""
    },

```

```
- 选择mimickit/run.py 按F5启动训练


















