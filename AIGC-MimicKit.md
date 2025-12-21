- 源代码地址 https://github.com/xbpeng/MimicKit?tab=readme-ov-file
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
- 选择mimickit/run.py 按F5启动训练
