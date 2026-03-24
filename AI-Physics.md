## FEM离散化和reduction
```
siggraph 2012 course notes

知识梳理：
 Model Reduction 数学公式： 常微分方程 u的2阶导数=F(u, u的1阶导数, t) -> 近似常微分方程 u = U q -> 变成q的一个常微分方程。
 Arpack solver 求特征值和特征向量
 1. Linear modal analysis reduction: M u的2阶导 + D u的1阶导 + K u = f， 模拟小变形
  M为mass， D为damping, K为stiffness
  Solve generalized eigenvalue problem ： A x = λ B x
 2. Model reduction of nonlinear deformations：Mu¨ + Du˙ + fint(u) = f .
 3.Applications of Model Reduction

Vega开源A free physics library to simulate 3D nonlinear deformable objects
 Linear FEM
 Co-rotational linear FEM
 Invertible FEM
 Saint-Venant Kirchhoff FEM
 Mass-spring Systems
```
## IK
```
剑桥Technical Report ： Inverse Kinematics: a review of existingtechniques and introduction of a new fast iterative solver


The joint angles are also written as a column vector θ = (θ1, ..., θn)
s = f (θ) This is called the Forward Kinematics (FK) solution
The goal of Inverse Kinematics (IK) is to find a vector θ such that ⃗s is equal to a given
desired configuration ⃗sd: θ = f^−1(sd)

方法：
  1.Jacobbi Inverse Methods
    Jacobian Pseudo-inverse
    Jacobian Transpose
    Singular Value Decomposition
    Damped Least Squares
    Pseudo-inverse Damped Least Squares
    Selectively Damped Least Squares
    Incorporating constraints
    Feedback Inverse Kinematics
2.Newton Methods
3.IK using Sequential Monte Carlo Methods 
4.Style or Mesh-based Inverse Kinematics
  Learning method to learning the meaningful space
5.Heuristic Inverse Kinematics Algorithms
  CCD Cyclic Coordinate Descent method
    Inductive Inverse Kinematic Algorithm
  Triangulation Inverse Kinematics
  Sequential Inverse Kinematics
6.FABRIK: A new heuristic IK methodology 
  FABRIK with Multiple End Effectors
  Applying Constraints to FABRIK
```

## 思考一个问题
```
目前Physics AI领域针对一个具体的操作任务，比如从vision sensor看到一个垃圾，它能自己寻找垃圾桶，机器人如何意识到要先走过去，绕开路上的椅子等遮挡物，或者跳过一个小凳子，然后使用一只手去捡起这个垃圾，再走到垃圾桶旁边，扔到垃圾桶里。 这里面可能设计到很多模块，目前前沿领域是怎么设计的，请推荐一些前沿的paper。我这边可能只关心如何再模拟世界而不关心sim-to-real。

一、技术架构的底层设计逻辑（仅面向模拟环境）
我将以「捡垃圾扔垃圾桶」任务为例，拆解前沿系统的模块化设计，所有方案均无需考虑真实世界的噪声、硬件误差等问题，完全适配仿真环境的特性：

🔍 多模态感知模块：从像素到语义的全链路理解
模拟环境中的感知不需要处理真实传感器的噪声，核心目标是用最小算力实现对环境要素的精准语义识别与空间定位，当前前沿方案已经形成了成熟的技术路径：

实例分割与语义对齐： 不再依赖传统的目标检测模型，而是直接用SAM（Segment Anything Model）的3D扩展版本，对仿真环境中的RGB-D点云做零样本实例分割，一次性识别出垃圾、垃圾桶、椅子、小凳子等所有目标物体，同时通过CLIP做视觉-语义对齐，让机器人理解每个物体的功能属性（比如「垃圾桶是用来装垃圾的」「椅子是障碍物需要绕开」）。 2026年最新的工作进一步优化了仿真场景下的感知效率，通过直接读取仿真引擎的底层语义标注，实现了对所有物体的100%识别准确率，省去了模型推理的算力开销。
物理属性感知： 直接从仿真引擎中读取物体的质量、摩擦系数、刚体约束等物理属性，无需通过视觉估计，为后续的抓取、运动规划提供精准的先验信息，这也是模拟环境相比真实世界的核心优势之一。
🧭 任务规划与运动决策：分层架构实现长周期任务推理
针对「移动-避障-抓取-扔垃圾」这样的长周期任务，前沿方案普遍采用三层分层规划架构，既保证了任务的逻辑正确性，又兼顾了运动的动态适应性：

高层任务拆解： 用大语言模型（LLM）作为「任务大脑」，将自然语言指令（如「捡起地上的垃圾扔进垃圾桶」）拆解为四步原子子任务：①移动到垃圾所在位置；②控制机械臂捡起垃圾；③移动到垃圾桶所在位置；④控制机械臂将垃圾扔进垃圾桶。 最新的进展是直接将VLA（视觉-语言-动作）模型作为高层规划器，输入仿真环境的视觉图像，直接输出子任务序列，无需人工设计任务模板。
中层运动规划： 针对「绕开椅子、跳过小凳子」的动态避障需求，前沿方案已经放弃了传统的A*、RRT等采样类规划算法，转而采用基于世界模型的预测式规划： 首先在仿真环境中训练一个动态世界模型，输入当前环境状态和机器人的候选动作，预测未来1-2秒的环境变化，判断动作是否会发生碰撞，是否能到达目标位置；机器人在虚拟空间中并行预演上百条可能的运动轨迹，选择最优的路径，不仅能绕开静态障碍物，还能应对动态移动的物体。 针对腿式机器人的「跳过小凳子」需求，采用模型预测控制（MPC）与强化学习结合的方案，在仿真中训练跳跃策略，通过世界模型预演跳跃的落地位置、重心变化，保证跳跃的稳定性。
底层动作生成： 直接输出机器人的关节控制指令（移动底盘的速度、机械臂的关节角度、抓取器的开合度），在仿真环境中完全不需要考虑硬件的延迟、误差，因此可以通过强化学习直接训练端到端的控制策略，实现动作的精准执行。
🦾 抓取与操作控制：仿真环境下的通用抓取策略
模拟环境中的抓取不需要考虑真实世界的接触噪声、物体形变误差，核心目标是实现对任意形状、重量物体的稳定抓取，当前前沿方案分为两类：

解析式抓取规划： 针对已知几何形状的物体，直接通过解析算法计算最优抓取点，结合仿真引擎的碰撞检测，判断抓取姿态是否可行，这种方法的速度极快，能在1ms内生成抓取指令，非常适合仿真环境中的大规模训练。
学习式抓取规划： 针对未知形状的物体，用GraspNet等模型基于点云数据预测抓取成功率，选择最优的抓取姿态，结合强化学习微调抓取力度，保证抓取的稳定性。2025年之后的工作进一步实现了抓取与全身运动的协同控制，机器人可以在移动的过程中同时调整机械臂姿态，大幅提升操作效率。
```
```
WholebodyVLA https://github.com/OpenDriveLab/WholebodyVLA
awesome-vision-language-action-model https://github.com/DelinQu/awesome-vision-language-action-model
awesome-vla-wam https://github.com/DravenALG/awesome-vla-wam
```


## [2025]PhysCtrl https://openreview.net/pdf?id=AHEKhff4Oa
有关工作 
- Neural Physical Dynamics
```
基于数值方法的传统物理模拟 FEM, PBD, MPM, SPH, mass-spring 系统

PINN: Physical Informed Neural Networks。用神经网络近似偏微分方程的解以及将物理约束放到神经网络里。缺点是：per-scene case，且论文实现的是流体。

ElastoGen：替换模拟中的一部分为神经网络以实现更快的推理。缺点：依赖Voxel Representation，只支持弹性材质，需要一个完全的3d模型作为输入。

GNN： 有好几篇论文，已经显示出能实现多种材质的模拟。缺点：依赖下一步动力学预测，容易累计误差或者漂移。

PhysCtrl： 使用点云表示 + spatial time trajectory diffusion model
```

- Controllable Video Generative Models
```
text-video对作为输入，训练视频生成模型。
也可把相机输入、人物姿势、点移动作为额外控制信号。


```



## MuJoCo https://github.com/google-deepmind/mujoco?tab=readme-ov-file
## opensim https://simtk.org/projects/opensim
## https://openaccess.thecvf.com/content/CVPR2024/papers/Ugrinovic_MultiPhys_Multi-Person_Physics-aware_3D_Motion_Estimation_CVPR_2024_paper.pdf
## https://openaccess.thecvf.com/content/CVPR2024/papers/Muller_Generative_Proxemics_A_Prior_for_3D_Social_Interaction_from_Images_CVPR_2024_paper.pdf
## Control Rig https://www.bilibili.com/video/BV1JT4y127z7/?spm_id_from=333.1387.search.video_card.click&vd_source=8cfcd5a041c19f2f5777e1a8d78359f2
## Starting From Chair https://www.ai.pku.edu.cn/en/info/1191/1948.htm
## Full-Body Articulated Human-Object Interaction https://arxiv.org/pdf/2212.10621
## Idea: Predict/Construct Pose from 3d target Object and prompt
## Predict Human pose from 3d scene data [2024]Diverse 3D Human Pose Generation in Scenes based on Decoupled Structure
## Physics Animation https://www.youtube.com/watch?v=46NfgXlnCzM

- 需要3D数据作为预测


## UE5 NNE
- UE5 Experimental Plugin NNERuntimeRDG
- UE5 Experimental Plugin NNERuntimeIREE
- UE5 Experimental Plugin NeuralRendering
- UE5 plugin Animation MLDeformer
- UE5 plugin AI MLAdapter
- UE5 plugin NNERuntimeORT
- UE5 plugin NNEDenoiser
- UE5 NNEEditor
- UE5 Engine Runtime NNE
```
UNNEModelData .onnx file is loaded into this object
given a UNNEModelData asset, the following interface is used to create a model

INNERuntime
INNERuntimeCPU
INNERuntimeGPU
INNERuntimeRDG

CreateModelCPU
```
- [PhysicsControl](https://dev.epicgames.com/documentation/en-us/unreal-engine/content-examples-sample-project-for-unreal-engine?application_version=5.4)
- Epic Game Launcher -> Unreal Engine -> Examples -> Content Example
- https://dev.epicgames.com/community/learning/talks-and-demos/qjj3/unreal-engine-new-character-physics-in-ue5-can-you-pet-the-dog-gdc-2023
- https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/Animation/PhysicsControl
- [UE物理引擎B站教程](https://www.bilibili.com/video/BV1ym421372T?spm_id_from=333.788.videopod.sections&vd_source=8cfcd5a041c19f2f5777e1a8d78359f2)
```
物理约束组件

```

## Physics Control

## ## hpp-core https://github.com/humanoid-path-planner/hpp-core/tree/master/include/hpp/core/problem-target




 
- [2025]Generative Physical AI in Vision: A Survey [Pdf Link](https://arxiv.org/pdf/2501.10928v2)
- [2022][ECCV]Transformer with Implicit Edges for Particle-based Physics Simulation
- [2024][sig asia]https://la.disneyresearch.com/wp-content/uploads/RobotMDM_red.pdf
```
kinematic motion generation
physics based character control
```
- 强化学习[2020] GDC https://www.youtube.com/watch?v=lN9pXZzR3Ys
- 
```
模拟了流体
```
- [2023]Towards Multi-Layered 3D Garments Animation
```
- [2024]Learning 3D Garment Animation from Trajectories of A Piece of Cloth
```
- [2025]GausSim: Foreseeing Reality by Gaussian Simulator for Elastic Objects
- How to model skeleton as XPBD simulation 
- https://github.com/emirsahin1/VRPhysicsCharacter/blob/main/VirtualCombatSim/Public/PhysicsHand.h
- PreintegratedSkinBRDF
- https://dev.epicgames.com/community/learning/courses/GAR/unreal-engine-learning-agents-5-5/1w7V/unreal-engine-improving-observations-debugging-5-5
- https://github.com/pau1o-hs/Learned-Motion-Matching/tree/master
- https://github.com/orangeduck/Motion-Matching/blob/main/resources/train_decompressor.py
- https://github.com/facebookresearch/metamotivo/edit/main/tutorial.ipynb
- Physics Control1 https://dev.epicgames.com/community/learning/tutorials/vrWZ/unreal-engine-physics-control-component-let-s-build-a-character-interactive-raft-in-the-water-subtitle
- Physics Control2 https://www.youtube.com/watch?v=iWjbPB25XUg
- https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Plugins/PhysicsControl/UPhysicsControlComponent
- https://dev.epicgames.com/community/learning/courses/GAR/unreal-engine-learning-agents-5-5/7dmy/unreal-engine-learning-to-drive-5-5
- lyra-sample-game-interaction-system https://dev.epicgames.com/documentation/en-us/unreal-engine/lyra-sample-game-interaction-system-in-unreal-engine?application_version=5.0
- https://www.youtube.com/watch?v=TnLhP81r45Q
- AI Controller + Behavior Tree + BlackBoard + MotionMatching. https://www.youtube.com/watch?v=sEKfLgWKdTg
- Lyra Interaction System https://dev.epicgames.com/documentation/en-us/unreal-engine/lyra-sample-game-interaction-system-in-unreal-engine?application_version=5.0
- Combat UEFN https://www.youtube.com/watch?app=desktop&v=HC3xa-laFTI
- https://github.com/sebastianstarke/AI4Animation


## Physics Papers
- Awesome-neural-physics https://hhuiwangg.github.io/projects/awesome-neural-physics/





