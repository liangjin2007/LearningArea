## Isaacgym Simulator
- Installation
```
git clone https://github.com/liangjin2007/ProtoMotions.git
sudo apt install git-lfs
git lfs fetch --all
conda create -n isaacgym python=3.8
conda activate isaacgym
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
pip install -e isaacgym/python
pip install -e /path/to/protomotions
pip install -r /path/to/protomotions/requirements_isaacgym.txt
```

- Data Preprocessing
```
See below

export PYTHONPATH="/home/liangjin/ProtoMotions:$PYTHONPATH"

python ./data/scripts/convert_amass_to_motionlib.py /home/liangjin/AMASS_npz/ /home/liangjin/AMASS_pt --motion-config data/yaml_files/amass_smpl_train.yaml --motion-config data/yaml_files/amass_smpl_test.yaml --motion-config data/yaml_files/amass_smpl_validation.yaml

```

- Train
```
train.sh
# To resolve isaacgym problem
export LD_LIBRARY_PATH=/home/liangjin/anaconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

export PYTHONPATH="/home/liangjin/ProtoMotions:/home/liangjin/newton:$PYTHONPATH"

python /home/liangjin/ProtoMotions/protomotions/train_agent.py \
    --robot-name smpl \
    --simulator isaacgym \
    --experiment-path /home/liangjin/ProtoMotions/examples/experiments/mimic/mlp.py \
    --experiment-name smpl_mlp_mimic \
    --motion-file /home/liangjin/AMASS_pt/amass_smpl_train.pt \
    --num-envs 512 \
    --batch-size 1024 \
    --ngpu 2 \
    --use-wandb
```


- Fix wandb
```
训练没有看到曲线信息

使用WSL尝试训练1个env：
    1. train.yaml太大，无法生成.pt文件，所以下面的launch.json中使用了test.pt
    2. num-envs设为1时，batch-size必须设为32
    3. num-envs设为1时会碰到python error需要修复几处文件中的错误 : 已提交到ue-simulator
        /protomotions/agents/utils/metering.py        
    4. 在train_agent.py的import wandb后添加:
        # 替换为你的WandB API密钥
        WANDB_API_KEY = "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"
        # 显式登录，跳过命令行输入
        wandb.login(key=WANDB_API_KEY)

ProtoMotions使用lightning这个基于Pytorch之上的分布式训练框架，写日志通过loggers作为参数传入lightning.Fabric的构造函数。
ProtoMotions写出训练日志是通过lightning.Fabric.log_dict(metrics_dict)写出。具体代码在 /protomotions/agents/base_agent/agent.py 804 行
    self.fabric.log_dict(aggregated_log_dict)
它有个问题： 没有提供step参数的值，导致曲线始终是在0上。
修改为：
    self.fabric.log_dict(aggregated_log_dict, step = self.step_count)




WSL训练的launch.json    
    {
        // Use IntelliSense to learn about possible attributes.
        // Hover to view descriptions of existing attributes.
        // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
        "version": "0.2.0",
        "configurations": [
    
            {
                "name": "Python Debugger: Current File",
                "type": "debugpy",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal",
                // parameters for python data/scripts/convert_amass_to_motionlib.py xxx
                // "args": [
                //     "/home/liangjin/Desktop/AMASS_npz/", "/home/liangjin/Desktop/AMASS_pt/",
                //     "--motion-config", "/home/liangjin/Desktop/ProtoMotions/data/yaml_files/amass_smpl_train.yaml"
                //     // "--motion-config", "/home/liangjin/Desktop/ProtoMotions/data/yaml_files/amass_smpl_test.yaml"
                //     //"--motion-config", "/home/liangjin/Desktop/ProtoMotions/data/yaml_files/amass_smpl_validation.yaml"
                // ],
                // parameters for training python protomotions/train_agent.py
                "args": [
                    "--robot-name", "smpl",
                    "--simulator", "isaacgym",
                    "--experiment-path", "/home/liangjin/Desktop/ProtoMotions/examples/experiments/mimic/mlp.py",
                    "--experiment-name", "smpl_mlp_mimic",
                    "--motion-file", "/home/liangjin/Desktop/AMASS_pt/amass_smpl_test.pt",
                    "--num-envs","1",
                    "--batch-size","32",
                    "--ngpu","1",
                    "--use-wandb"
                ],    
                "env": {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "LD_LIBRARY_PATH": "/usr/lib/wsl/lib:\"${env:CONDA_PREFIX}/lib\":\"${env:LD_LIBRARY_PATH}\""
                },
            }
        ]
    }


```


- Inferece
```
Ref https://protomotions.github.io/tutorials/workflows/amass_smpl.html, part Evaluation
https://protomotions.github.io/user_guide/experiments.html

Run inference on trained model:
python protomotions/inference_agent.py \
    --checkpoint results/smpl_amass_flat/last.ckpt \
    --simulator isaacgym


evaluate.sh
# To resolve isaacgym problem
export LD_LIBRARY_PATH=/home/liangjin/anaconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

export PYTHONPATH="/home/liangjin/ProtoMotions:/home/liangjin/newton:$PYTHONPATH"

python /home/liangjin/ProtoMotions/protomotions/inference_agent.py \
    --simulator isaacgym \
    --checkpoint /home/liangjin/ProtoMotions/results/smpl_mlp_mimic/last.ckpt \
    --motion-file /home/liangjin/AMASS_pt/amass_smpl_test.pt



Full evaluation over all motions:

python protomotions/inference_agent.py \
    --checkpoint results/smpl_amass_flat/last.ckpt \
    --simulator isaacgym \
    --num-envs 1024 \
    --full-eval

```















