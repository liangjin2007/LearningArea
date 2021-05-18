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
- [2020]RigNet： GNN方法， 输出Skeleton and skinning， LBS basedrig.
