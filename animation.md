#### animation表示方法

- [1988]LBS : Linear blend skinning
- [2007]DQS : Dual Quaternion Skinning
- [2020]Deep Learning： 直接预测顶点位置
- blend shapes : joint rotation之后再连blend shape系数， 比如[2015]
- SMPL: 要求输入网格拓扑一致
- STAR


#### Mesh deformation of articulated shapes
- LBS
- DQS
- Spherical based skinning
- Multi-linear techniques
- cages

- corrective deformations
  - example-based methods
    - 一般将例子encoding为blendshape
  - vertex displacements
  - statistical models
  - compact sparse format
  - RBF-based blend space
  - Bone -> blendshapes, constrained geometric optimization problems
  - dynamic simulation




#### Automatic skinning and rigging

传统优化方法
- [2007]projection and heat diffusion
- [2011]bounded biharmonic energy
- [2013]geodesic voxel binding
- [2012]physics-inspired approaches

深度学习方法
- [2019]NeuroSkinning: GNN方法
- [2020]RigNet： GNN方法， 输出Skeleton and skinning， LBS based rig.


#### locomotion
- 跟motion意思很接近，只是表示从一个地方到另一个地方的移动。 比如走，跑，跳等。
- motion > locomotion

#### motion editing techniques
- layering
- slicing
- optimization: e.g. ik, impose kinematic and dynamic constraints
- deep learning techniques

#### motion synthesis by Interpolation
- generally need align animation along the timeline
- spatial constraints
- data-driven motion synthesis
  - Gausssian Processes(GP) 

#### character control


#### Papers
[2021]Neural Animation Layering for Synthesizing Martial Arts Movements


