# 市面上的头发模拟产品
- nHair
- xGen
- UE GroomActor https://www.fxguide.com/fxfeatured/ue4-real-time-hair-advances/

- UE 头发介绍https://docs.unrealengine.com/en-US/Engine/HairRendering/index.html
```
DCC创建Groom并导出Alembic
将Alembic导入UE4
为骨架网格体创建绑定资产
将Groom资产置于世界场景中（作为蓝图中的Actor）
使用转移蒙皮将Groom组件连接到骨骼网格体socket
设置并指定毛发材质
Details -> Material -> Shading Model设为Hair。
用途类别下启用Use with Hair Strands
毛发属性材质表达式示例
```

- Niagara https://www.youtube.com/watch?v=31GXFW-MgQk
```
Niagara Visual Effects Documentation-
https://docs.unrealengine.com/Engine/Niagara/index.html

Vertex Animation Tool -
https://docs.unrealengine.com/en-US/Engine/Animation/Tools/VertexAnimationTool/index.html

Further, Inside Unreal videos you might enjoy: 


An Introduction To Niagara | Feature Highlight | Unreal Engine Livestream-
https://youtu.be/s7QpAkzx5eM

1.World Interaction
Triangles，Physics Volumes, Scene Depth, Distance Fields, volume textures
Distance Fields
三个使用方法：查询距离，计算梯度方向，递归查询从某个Particle开始的某个方向上的最近三角形。


```

#Taichi GAMES201
## Mass-Spring System
## SPH : 基于例子的模拟， 全名光滑例子流体
- SPH
- PCI-SPH: Predictive-Corrective Incompressible SPH
- PBF: Position-based fluids ti example pbf2d
- Survey Paper: SPH fluids in Computer Graphics
- CFL condition
- Accelerating SPH: Neighborhood Search
  - voxel grid
  - Compact Hashing
- Discrete element method
- MPS: Moving Particle Semi-implicit
- Power Particles: an incompressible fluid solver based on power diagrams
- A peridynamic perspective on spring-mass fracture
## 变形、弹性和有限元
- 弹性材质elastic
- 弹性材质是其他材质的基础 viscoelastic（粘弹性）, elastoplastic(弹性塑料), viscoplastic(粘塑性的)
- FEM method: [2012][sig course]FEM simulation of 3D deformable solids: a practitioner’s guide to theory, discretization and model reduction
- MPM method: [2016]The material point method for simulating continuum materials
- Deformation: 变形
- 超弹性hyperelasticity
  - stress-strain relationship
  


- https://taichi.readthedocs.io/zh_CN/latest/scalar_tensor.html

# MPM
- MPM论文列表 https://www.seas.upenn.edu/~cffjiang/mpm.html
- physics based animation http://www.physicsbasedanimation.com/resources-courses/

## Problems
large deformation
mesh distortion
fracture
self-collision
coupling between materials

## Simulation Objects
elastic objects
snow
lava
sand
viscoelastic fluids
Production pipline Disney
cloth
thin shell
hair

## 运动学概念Concepts
- 连续介质运动
- 变形
  - 变形映射
  - 变形梯度Deformation gradient F，代表材质局部变形
  - 变形梯度的行列式,代表体积的局部变化。行列式大于1表示体积增加，行列式小于1表示体积减小。J=0意味着体积变为0。J < 0意味着材质翻转。
- 往前推和往后拉
  - 拉格朗日视角：把quantities看成是（X, t）的函数。变形映射是个双射。
  


## Methods
- Lagrangian **FEM**
- hybrid Eulerian/Lagrangian Material Point Method(**MPM**)
- Particle Method: Smoothed Particle Hydrodynamics(**SPH**)
- Poisson Based Dynamics(**PBD**)


## MPM算法explicit integration

## MPM算法implicit integration

## 参考
[1995]Particle in Cell

# FEM Simulation of 3D Deformable Solids比如果冻
http://www.physicsbasedanimation.com/2012/08/15/siggraph-course-fem-simulation-of-3d-deformable-solids-a-practitioners-guide-to-theory-discretization-and-model-reduction/

# Hair Modeling, Simulation and Rendering
https://hal.inria.fr/inria-00520193/document


# 动画
- https://deepmotionediting.github.io/retargeting
- https://deepmotionediting.github.io/style_transfer
- MeshCNN https://ranahanocka.github.io/MeshCNN/

# position-based dynamics framework， 基于物理的动画
[2017]A Survey on Position Based Dynamics, 2017
- 不是很准确，但是可用于交互场景
- 可以模拟弹性杆，布料，体变形体，刚体和流体。

[2006]Positino Based Dynamics
- 力
- 冲量
- 动量
- 加速度
- 速度
- 质量
- 密度
- 是否保存速度。不保存速度，速度是通过前后两次的位置隐式表示。
- PBD表示方法


# force-based method


# 头发模拟
- low bending
- shear resistance阻力
- stretch resistance阻力
- collision碰撞
- friction摩擦
- electrostatic forces
- hair product

- https://www.fxguide.com/fxfeatured/ue4-real-time-hair-advances/


# 布料模拟
- 【UE4布料】7分钟内在虚幻引擎中进行布料模拟 https://www.bilibili.com/s/video/BV1At4y1D7WQ
- 流程十：UE4为衣服添加布料、胸部头发添加物理效果（完结）https://space.bilibili.com/48806445?spm_id_from=333.905.b_7570496e666f.3


# PBF
```
Enforce constaint density 保持密度不变
C(p1, p2, ..., pn) = xxx

particle clustering or clumping

surface tension-like effects

Vorticity Confinement


CFM constraint force mixing
```

# 风模拟
- NvCloth/docs/documentation/Solver/Index.html#wind-simulation




