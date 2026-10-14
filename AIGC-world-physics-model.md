## 目录
- [1.Survey](#1Survey)
- [2.Rigid_Deformable_Articulated_Object_Generation](#2Rigid_Deformable_Articulated_Object_Generation)
- [3.SceneGeneration](#3SceneGeneration)
- [4.SimulationPlatform](#4SimulationPlatform)
## 1.Survey
- [2026][Robotics:Science and Systems]SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation
  - 单臂+灵巧手抓工具benchmark 
  - github链接 https://github.com/tylerlum/simtoolreal, 398 star, 有预训练模型可下载以及有训练代码

- [2025]GraspGen: A Diffusion-based Framework for 6-DOF Grasping with On-Generator Training
  - 实时Grasp预测，20HZ推理
  - Diffusion based

- [2025][NeurIPS]PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation
  - https://github.com/cwchenwang/physctrl, star 130, star数不多，看来这个文章估计没啥实际价值

- [2025][ICCV]PhysSplat:
  - 无代码 

- [2024]PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics
  - integrates physically grounded Newtonian dynamics within 3D Gaussians



## 2.Rigid_Deformable_Articulated_Object_Generation
- [2022][CVPR]Ditto: Building Digital Twins of Articulated Objects from Interaction
  - https://github.com/UT-Austin-RPL/Ditto
  - before, after point cloud, estimate articulator 位置
- [2023][CVPR]Infinite Photorealistic Worlds using Procedural Generation
  - https://github.com/princeton-vl/infinigen, princeton, 7.3k star, updated last month.
- [2024][CVPR]
  - https://github.com/3dlg-hcvc/cage
  - DDPM based
  - Graph structure to describe articulated 3d objects
  - Node attributes
    - Bounding box
    - Joint Type
    - Joint Axis
    - Joint Range
    - Semantic Label
- [2024][RSS]URDFormer
  - https://github.com/WEIRDLabUW/urdformer
- PhysX-Omni: Unified Simulation-Ready Physical 3D Generation for Rigid, Deformable, and Articulated Objects
  - https://github.com/physx-omni/PhysX-Omni, 319 star
- [2026]URDF-Anything+:End-to-End Generation for Simulation-Ready Articulated Assets
  - https://github.com/URDF-Anything-plus/Code, 53 star
- [2026][CVPR]PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image
  - https://github.com/ziangcao0312/PhysX-Anything, 945 star
- Dataset
  - PhysXNet, PhysX-Mobility, and PhysXVerse

- [Motion3-to-4](https://github.com/Inception3D/Motion324) 
## 3.SceneGeneration
- [2026][ICLR]ReconViaGen: Towards Accurate Multi-view 3D Object Reconstruction via Generation
  - https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen

## 4.SimulationPlatform
- [2025]RoboVerse
  - https://github.com/RoboVerseOrg/RoboVerse， 1.9k star
- []FEBio
  - https://github.com/febiosoftware/FEBio
  - CPU代码
- []artisynth
  - https://github.com/artisynth/artisynth_core
  - Java写的 
- []SOFA
  - https://github.com/sofa-framework/sofa, 1.3k star
  - 有基于CUDA的代码
- []HAELE 3D is a family of interactive 3D anatomy and drawing reference tools for artists, available on Steam.
  Key Software in the Series
  • HAELE 3D - Pose Studio Max: A comprehensive full-body anatomy and gesture reference tool featuring realistic and simplified models, skeletal/muscular systems, props, and VR mode support.
  • HAELE 3D - Portrait Studio Pro: Specialized in human faces, heads, expressions (90+ expressions), and cranial anatomy.
  • HAELE 3D - Hand Poses / Feet Poses: Focused tools for detailed hand and foot articulation, skin tones, and texture adjustments.
