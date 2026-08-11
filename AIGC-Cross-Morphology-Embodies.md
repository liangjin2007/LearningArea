# 主题
本文档主要想调研当下基于IsaacLab/IsaacSim/Genesis/Genesis-world的Cross-Morphology 具身智能算法。

其中Cross-Morphology指的是一个模型可以用做不同机器人的策略(Policy)模型。

**目录**
- [0.Python基础](#0.Python基础)
- [1.IsaacLab安装](#1.IsaacLab安装)

## 0.Python基础
- @dataclass decorator
- @configclass decorator


## 1.IsaacLab安装
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
```



## 2. IsaacLab Generate Your Own Project or Task
### 2.1.isaaclab.bat --new
```
  isaaclab.bat --new
```
- Project Structure
![Project Structure](https://isaac-sim.github.io/IsaacLab/main/_images/walkthrough_project_setup.svg)

### 2.2.安装和Run
```
cd path
python -m pip install -e source/<given-project-name>
```

### 2.3. scripts/list_envs.py
列出项目中的可行任务
```
python scripts\list_envs.py
可手工修改，如果任务名修改了的画。
```

### 2.4. Run a task
```
python scripts\<specific-rl-library>\train.py --task=<Task-Name>
```

### 2.5. Run a task with dummy agents
用来确认环境是否正确配置
- Zero-action agent
```
python scripts\zero_agent.py --task=<Task-Name>
```
- Random-action agent
```
python scripts\random_agent.py --task=<Task-Name>
```

### 2.6. Environment Design Background
- ![App, Simulation, World, Stage, and Scene](https://isaac-sim.github.io/IsaacLab/main/_images/walkthrough_sim_stage_scene.svg)
```
World指空间坐标系和单位
App和Sim是Above World
Stage和Scene是Below World的概念
Stage提供给World组合上下文。类似于带变换的节点层级。
Scene
```
- ![Stage Example](https://isaac-sim.github.io/IsaacLab/main/_images/walkthrough_stage_context.svg)

### 2.7.Task specific Code
- xxx_env_cfg.py
```
默认生成的是CARTPOLE任务
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class TraincrossmorphologyEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    # - action scale
    action_scale = 100.0  # [N]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]
```
- xxx_env.py
```
# imports
.
.
.
from .isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg

class IsaacLabTutorialEnv(DirectRLEnv):
    cfg: IsaacLabTutorialEnvCfg

    def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        . . .

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        . . .

    def _apply_action(self) -> None:
        . . .

    def _get_observations(self) -> dict:
        . . .

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(...)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        . . .

    def _reset_idx(self, env_ids: Sequence[int] | None):
        . . .

@torch.jit.script
def compute_rewards(...):
    . . .
    return total_reward
```

### 2.8. Environment Design
```
Define the robot
  jetbot.py ArticulationCfg(spawn=xxx, actuators={xxx})

Environment Configuration
  @configclass
  class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):
      # env
      decimation = 2
      episode_length_s = 5.0
      # - spaces definition
      action_space = 2
      observation_space = 3
      state_space = 0
      # simulation
      sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
      # robot(s)
      robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
      # scene
      scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)
      dof_names = ["left_wheel_joint", "right_wheel_joint"]
Define the training simulation and manage cloning

Apply the actions from the agent to the robot

Calculate and return the rewards and observations

Manage resetting and terminal states
```





## 3.IsaacLab tutorials
### 3.1.Creating empty scene
- Launching the simulator
```
from isaaclab.app import AppLauncher
import argparse
# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

- Importing python modules
```
# after simulation app is running, we can import various python modules.

from isaaclab.sim import SimulationCfg, SimulationContext
```

- Configuring the simulation context
```
# 其中isaaclab.sim其实继承自isaacsim isaacsim.core.api.simulation_context.SimulationContext

```














https://github.com/UMass-Embodied-AGI/Genesis-Humanoid
