# 主题
本文档主要想调研当下基于IsaacLab/IsaacSim/Genesis/Genesis-world的Cross-Morphology 具身智能算法。

其中Cross-Morphology指的是一个模型可以用做不同机器人的策略(Policy)模型。

**目录**
- [1.embodiment-scaling-laws](#1.embodiment-scaling-laws)

## 1.embodiment-scaling-laws
- 链接 https://github.com/BoAi01/embodiment-scaling-laws
```
1.1. 安装IsaacLab https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
  Enable long path support : HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled
  下载anaconda3
  conda create -n env_isaaclab python=3.11
  conda activate env_isaaclab
  python -m pip install --upgrade pip # windows or pip install --upgrade pip on linux
  Installing dependencies
    Install Isaac Sim pip packages: pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
    Install a CUDA-enabled PyTorch build that matches your system architecture:pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```




https://github.com/UMass-Embodied-AGI/Genesis-Humanoid
