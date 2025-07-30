# About how to encoding Motion Matching Into Networks and how to apply into UE.

## Reference
- dataset https://github.com/ubisoft/ubisoft-laforge-animation-dataset
- https://github.com/pau1o-hs/Learned-Motion-Matching/tree/master
- https://github.com/orangeduck/Motion-Matching/blob/main/resources/train_decompressor.py
- Robust Motion Inbetween (LSTM based) Robust motion in-betweening
- SIG 2024 https://github.com/setarehc/diffusion-motion-inbetweening

## 目标
- 避免滑步：Runtime节点中只要脚是着地的必须不移动。整个运动的移动应该交给RootMotion来控制。
- 快速牵引角色的各个部位到目标
  - Motion Matching模型预测出Pose。
- 约束：
  - 按照角色的关节物理约束运动
  - 自己维护物理模拟状态求解物理资产中的胶囊体的碰撞约束
