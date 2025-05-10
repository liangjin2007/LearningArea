## Basics 
- Animation Blueprint
  - State Machine
  - Behavior Tree
  - Blackboard
  - Anim Nodes
    - Copy pose from mesh
    - Retarget
    - Modify Curve
    - Modify Bones
    - IK Rig
    - Control Rig
    - Blend Poses per bone
    - Layered Blend poses
    - State pose cache
    - Slot
- Animation Sequence
- Animation Montage
- Todo
  - Animation Modifier
  - Animation Layer
  - Animation Layer Interface
  - Motion Warp
  - Animation Warp

## Level Sequence Asset
双击该资产会出现Sequencer、Anim Outliner、Sequencer Curves、Animation窗口

## Control Rig
- 选择骨架层次中某个或某些骨骼后，可创建Bone, Control, Control Null。
- Control翻译为控制点。
- 控制点用颜色区分左边（绿色）、中间（黄色）、右边（红色）



## Anim Outliner
列出的是多个Control Rig




## Sequencer
- 可添加骨骼网格体、Actor、LevelSequencer等(可从内容浏览器中拖入到Sequencer)
- 可添加当前关卡中的物体
- 可添加Animation
- 可添加Control Rig
- 添加关键帧，按Enter键。  选择一条曲线，比如Transform -> Location -> X, 修改值（可以在关卡中查看变化）
