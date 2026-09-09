# genesis-world

看起来比isaaclab设计得简单很多，代码往往能在一个文件中写完；同时它支持各种各样的模拟器，似乎更贴近我们的需求。

## 文档
- https://genesis-world.readthedocs.io/en/latest/user_guide/overview/index.html

## 安装
```
想着可能会定制自己的模拟器，将它9月9日的代码作为基础clone到本地，在它代码的基础上去改出符合我们需求的Demo。
0. conda create -n "genesis" python=3.10

1.安装pytorch
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

2.安装genesis-world
  pip install genesis-world
  git clone https://github.com/Genesis-Embodied-AI/genesis-world.git
  cd genesis-world
  pip install -e ".[dev]"

3.安装额外
IPC solver (uipc backend)	:
  pip install pyuipc (Linux / Windows x86, NVIDIA GPU)
Nyx renderer:
  pip install gs-nyx
```
## HelloWorld
```
修改examples/tutorials/hello_genesis.py
修改2处：显示viewer和后端使用gpu.
gs.init(backend=gs.gpu)

scene = gs.Scene(show_viewer=True) # 默认不显示viewer
scene.add_entity(...)
scene.add_entity(...)

# scene.build() # n_envs=0
scene.build(n_envs=100, env_spacing=(2.5, 2.5), n_envs_per_row=10) # 后两个参数中的env_spacing必须设置，否则100只机械臂显示在一个地方。


for i in range(1000):
    scene.step()

然后执行python examples/tutorials/hello_genesis.py
可看到一只机械臂受重力落下。
```

## 概念和Configuration
### 核心概念
- Object Model
```
Entity
  Morph: Cube, Sphere, Mesh, URDF, MJCF, ...
  Material: Rigid, MPM(Elastic, Elasto-plastic, Liquid), PBD(Rode, Cloth, Volumetric), FEM(Elastic, Muscle, Thin Shell), SPH(Liquid, etc), ...
  Surface: Plastic, Rough, Glass, Metal, Copper, ...

Scene
  由Entity组成
  Simulator
    RigidSolver
      AvatarSolver
    MPMSolver
    PBDSolver
    FEMSolver
    SPHSolver
    SFSolver
    ...
    Coupler
```


- Local and global indexing
```
struct_particle_state_render = qd.types.struct(
    pos=gs.qd_vec3,
    vel=gs.qd_vec3,
    active=gs.qd_bool,
)

self.particles_render = struct_particle_state_render.field(
    shape=(self._n_particles, self._B),  # every particle in the scene, across all envs
    needs_grad=False,
    layout=qd.Layout.SOA,
)

pos = rigid_entity.get_dofs_position(dofs_idx_local=[2])  # shape ([n_envs,] 1)
tgt = entity.get_dofs_position()  # shape ([n_envs,] entity.n_dofs)
```
### Initialization and backends
```
backends
  gs.gpu, gs.cpu, gs.metal, ...
  print(gs.device)
  gs.init
  gs.destroy()
Precision
  init(precision="32")

可复现相关
  seed=0, debug=True

Deterministic mode
  use_deterministic_algorithms=True

logging
  logging_level="warning"

performance mode
  With performance_mode=True, the compiler bakes static tensor shapes into its kernels for roughly 30% faster simulation, at the cost of recompiling whenever the scene changes (which can take several minutes). Leave it off for research, debugging, and interactive work; turn it on for policy training and production runs where the scene is fixed.
```
### Options System 选项系统
```
gs.options.Options
  gs.options.SimOptions
  gs.options.RigidOptions
  gs.options.ViewerOptions

关于SimOptions和SolverOptions
SimOptions holds settings that are global by default: most importantly the timestep dt (seconds) and gravity (m/s², pointing down -Z). Each solver also exposes those same settings on its own options object, where they default to None.

A value set on a solver’s options overrides the global SimOptions value, for that solver only, and a solver whose field is left at None inherits the global value.

SimOptions.dt is how much simulated time one scene.step() advances. A solver’s dt is the interval it integrates over, so it must divide the step a whole number of times, and that quotient is the number of substeps per step. Every active solver advances together, so the count one solver asks for is the count they all take, and two solvers asking for different intervals raise. SimOptions.substeps requests the same count directly, and setting both raises unless they agree.

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),        # one step advances 0.01 s
    rigid_options=gs.options.RigidOptions(dt=0.005),   # two substeps per step, for every solver
    # mpm_options left unset -> the MPM solver, if used, integrates twice per step as well, over 0.005 s
)

The same inheritance applies to gravity, and each solver keeps its own value, so read it back from the solver simulating the entity, for example scene.rigid_solver.get_gravity(envs_idx). A scene coupling its solvers through IPC applies the SimOptions gravity to every body it couples, so a coupled solver authoring a different one raises at build time. Settings that are meaningful only to one solver (for example RigidOptions.constraint_solver or RigidOptions.max_collision_pairs) live solely on that solver’s options and have no global counterpart.

```
- Per Entity options
```
Per-entity options
add_entity takes its own options describing a single entity rather than the scene:

franka = scene.add_entity(
    morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    material=gs.materials.Rigid(),
    surface=gs.surfaces.Default(),
)
Morph: the entity’s geometry and initial pose. See Hello, Genesis World for loading morphs and the morph API.

Material: how the entity responds to physical forces, and which solver simulates it. See Beyond rigid bodies.

Surface: how the entity looks when rendered. See Surfaces and textures.
```

### 惯例Conventions
- Coordinate system
```
We use a right-handed, Z-up coordinate system. Relative to the default viewer, whose camera sits on the +X side looking back toward the origin:

+X: points out of the screen, toward the viewer.

+Y: points to the viewer’s right.

+Z: points up.
```
- Quaternion
```
w, x, y, z
Euler angles, where they are accepted instead, are extrinsic x-y-z in degrees (SciPy’s convention).
```
- Gravity
```
（0， 0， -9.81）， 9.81m/s^2
```
- Units
```
Units
Genesis World is unitless in the sense that it does no conversion for you, but every built-in default is expressed in SI units, and the API assumes you follow suit:

Length in meters, mass in kilograms, time in seconds.

Angles in radians, with one deliberate exception: Euler angles passed to morphs are in degrees (see the rotation section above).

Derived quantities follow from these: density in kg/m³, force in newtons, gravitational acceleration in m/s².

The simulation timestep is a duration in seconds, defaulting to dt = 1e-2 (10 ms):

gs.options.SimOptions(dt=0.01)
```
### Tensor shapes and batching
```
Genesis World simulates many environments in parallel (see Parallel simulation), so most quantities carry an optional leading batch dimension. The docs and docstrings describe shapes with a bracket notation:

distances  # shape ([n_envs,] n_probes)
points     # shape ([n_envs,] n_probes, 3)
The [n_envs,] bracket means: present when the scene is built with multiple environments, absent otherwise. A scene built with scene.build(n_envs=4096) returns tensors with a leading 4096 dimension; a scene built without n_envs drops that dimension entirely rather than using a size-1 axis.

Methods that read or write per-environment state take an envs_idx argument to address a subset of environments. Passing envs_idx=None (the default) applies to all of them; passing a tensor of indices selects only those rows along the batch dimension.
```
### Data types and precision
```

```
### Checkpoints and simulation states
State model
```

```
## Integrate MuscleMimic's fullbody
```
看MuscleMimic的代码musclemimic\environments\humanoids\myofullbody.py, 它是用python调用mujoco package的API去创建一个全身的MJCF出来。
那一个想法是把MuscleMimic作为依赖包放到genesis-world中，看看能否写转换代码，变成genesis-world中的scene。


```
