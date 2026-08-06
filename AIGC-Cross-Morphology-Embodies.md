# 主题
本文档主要想调研当下基于IsaacLab/IsaacSim/Genesis/Genesis-world的Cross-Morphology 具身智能算法。

其中Cross-Morphology指的是一个模型可以用做不同机器人的策略(Policy)模型。

**目录**
- [1.IsaacLab安装](#1.IsaacLab安装)

## 1.IsaacLab安装
- 链接 https://github.com/BoAi01/embodiment-scaling-laws
```
1.1. 安装IsaacSim https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
  Enable long path support : HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled
  下载anaconda3
  conda create -n env_isaaclab python=3.11
  conda activate env_isaaclab
  python -m pip install --upgrade pip # windows or pip install --upgrade pip on linux
  Installing dependencies
    Install Isaac Sim pip packages: pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
    Install a CUDA-enabled PyTorch build that matches your system architecture:pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

1.2.安装IsaacLab
  git clone https://github.com/isaac-sim/IsaacLab.git --branch main
  cd IsaacLab
  安装依赖
  isaaclab.bat --help
    usage: isaaclab.bat [-h] [-i] [-f] [-p] [-s] [-v] [-d] [-n] [-c] -- Utility to manage Isaac Lab.
    
    optional arguments:
       -h, --help           Display the help content.
       -i, --install [LIB]  Install the extensions inside Isaac Lab and learning frameworks (rl_games, rsl_rl, sb3, skrl) as extra dependencies. Default is 'all'.
       -f, --format         Run pre-commit to format the code and check lints.
       -p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active).
       -s, --sim            Run the simulator executable (isaac-sim.bat) provided by Isaac Sim.
       -t, --test           Run all python pytest tests.
       -v, --vscode         Generate the VSCode settings file from template.
       -d, --docs           Build the documentation from source using sphinx.
       -n, --new            Create a new external project or internal task from template.
       -c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'env_isaaclab'.
       -u, --uv [NAME]      Create the uv environment for Isaac Lab. Default name is 'env_isaaclab'.

  安装isaaclab依赖
    打开Anaconda powershell prompt
    conda activate env_isaaclab
    isaaclab.bat --install :: or "isaaclab.bat -i"
  Verifying the Isaac Lab installation
    :: Option 1: Using the isaaclab.bat executable
    :: note: this works for both the bundled python and the virtual environment
    isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py

    :: Option 2: Using python in your virtual environment
    python scripts\tutorials\00_sim\create_empty.py

    // 问题1： xxx.dll问题
    //   降级h5py： python.exe -m pip install "h5py==3.15.1"

    // 问题2： rxxXXX crash.
    // 固有问题。 解决办法：驱动 595.79太高，需要降级为591.74
    // 从https://www.nvidia.cn/geforce/drivers/details/260464/ 下载591.74驱动。
    
    再用isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py 就能启动IsaacSim 5.1.0窗口。

    Asset Caching: isaaclab.bat -s

```

## 2. IsaacLab 
```
Generate Your Own Project
  isaaclab.bat --new
    选External
    ...
    选path
    选name
安装和Run
  cd path
  python -m pip install -e source/<given-project-name>
Configurations
Robots
Apps and Sims
  a Standalone app

Create new project or task: https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html
```
- Project Structure
![Project Structure](https://isaac-sim.github.io/IsaacLab/main/_images/walkthrough_project_setup.svg)

https://github.com/UMass-Embodied-AGI/Genesis-Humanoid
