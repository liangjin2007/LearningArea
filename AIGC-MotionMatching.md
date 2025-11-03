# About how to encoding Motion Matching Into Networks and how to apply into UE.

## Reference
- dataset
  - ubisoft-laforge dataset https://github.com/ubisoft/ubisoft-laforge-animation-dataset
  - HumanML3D dataset https://github.com/EricGuo5513/HumanML3D
  - Kit dataset https://motion-annotation.humanoids.kit.edu/dataset/
  - How to generate wanted dataset https://github.com/zju3dv/EasyMocap
  - https://github.com/IDEA-Research/Motion-X
- https://github.com/pau1o-hs/Learned-Motion-Matching/tree/master
- https://github.com/orangeduck/Motion-Matching/blob/main/resources/train_decompressor.py
- Combining Motion Matching and Orientation Prediction to Animate Avatars for Consumer-Grade VR Devices
- Robust Motion Inbetween (LSTM based) Robust motion in-betweening
- SIG 2024 https://github.com/setarehc/diffusion-motion-inbetweening
- simulator
  - c++ human simulator https://github.com/google-deepmind/mujoco?tab=readme-ov-file
  - https://github.com/HoangGiang93/URoboViz?tab=readme-ov-file
  - MuJoCo-Unreal-Engine-Plugin https://lab.uwa4d.com/lab/67e6900f333affa84f3e3bac
  - XCCQuinn
  - mujoco_mpc real-time behaviour synthesis with MuJoCo, using Predictive Control https://github.com/google-deepmind/mujoco_mpc
  - A collection of high-quality models for the MuJoCo physics engine, curated by Google DeepMind. https://github.com/google-deepmind/mujoco_menagerie
  - Imitation learning benchmark focusing on complex locomotion tasks using MuJoCo  https://github.com/robfiras/loco-mujoco
## 目标
- 避免滑步：Runtime节点中只要脚是着地的必须不移动。整个运动的移动应该交给RootMotion来控制。
  - 其他参考链接： https://dev.epicgames.com/documentation/en-us/unreal-engine/fix-foot-sliding-with-ik-retargeter-in-unreal-engine
- 快速牵引角色的各个部位到目标
  - Motion Matching模型预测出Pose。
- 约束：
  - 按照角色的关节物理约束运动
  - 自己维护物理模拟状态求解物理资产中的胶囊体的碰撞约束



- Metahuman
```
https://dev.epicgames.com/community/learning/tutorials/GDMx/metahuman-animation-tutorial-series
https://www.bilibili.com/video/BV1E4MNzUEat?spm_id_from=333.788.player.switch&vd_source=38a71595005700b6ff304d0a48055f82
https://dev.epicgames.com/community/learning/tutorials/B3Yy/metahuman-living-notebook-ue5-6?locale=zh-cn
https://dev.epicgames.com/documentation/zh-cn/metahuman/metahuman-5-6-release-notes
```
