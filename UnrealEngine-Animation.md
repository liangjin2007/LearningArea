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
- Animation ChooserTable
  - 创建 Animation -> 选择器表 -> 设置动画选择器和Anim Instance类型 
  - 设置到动画蓝图的Sequence Player中的细节Update时使用Evaluate Chooser和Set Sequence With Inertial Blending从选择器表获取动画资源并设置到Sequence Player中。
- Todo
  - Animation Modifier
  - Animation Layer
  - Animation Layer Interface
  - Motion Warp
  - Animation Warp

## Level Sequence Asset
- 双击该资产会出现Sequencer、Anim Outliner、Sequencer Curves、Animation窗口。
- Sequencer Curves窗口如果没出现，可以在Sequencer的工具栏按钮中找到。


## Control Rig
- 选择骨架层次中某个或某些骨骼后，可创建Bone, Control, Control Null。
- Control翻译为控制点。
- 控制点用颜色区分左边（绿色）、中间（黄色）、右边（红色）
- C++类
```
UControlRigComponent
UControlRig
UControlRigBlueprint: Control Rig资产继承自这个类。
```
- Rig Inversion: Rig inversion enables users to remap existing animation onto animation rigs
https://dev.epicgames.com/documentation/en-us/unreal-engine/control-rig-inversion?application_version=4.27


## Anim Outliner
列出的是多个Control Rig

## PhysicsControl
- Body Modifier
```
Actor:
  BP_PhysicsBox
    Cube： root component
      PhysicsControl:  Physics Control Component

事件图表
   PhysicsCoontrol -> CreateBodyModifier(TargetComponent, BoneName, Set, BodyModifierDataCollisionType::Kinematic, BodyModifyDataCollisionType, BodyModifyDataGravityMultiplier， blend weight, use skeletal animation, update kinematic from simulation)
  PhysicsCoontrol -> SetBodyModifierMovementType
  PhysicsCoontrol -> SetBodyModifierKinematicTarget
```

- World-space Control
```
Actor:
  BP_PhysicsControlBox
    Cube： root component
      PhysicsControl:  Physics Control Component

PhysicsControlData结构体
  Linear Strength
  Linear Damping Ratio
  Linear Extra Damping
  Max Force
  Angular Strength
  Angular Damping Ratio
  Angular Extra Damping
  Max Torque
  Enabled
  Disable Collision
  Use Custom Control Point
  Use Skeltal Animation
  Only Control Child Object
  Linear Target  Velocity Multiplier
  Skeletal Animation Velocity Multiplier
  Angular Target Velocity Multiplier


事件图表
   PhysicsCoontrol -> CreateControl(ParentComponent, ParentBoneName, ChildComponent, ChildBoneName, ControlData, ControlTarget, Set, NamePrefix)
  PhysicsCoontrol -> SetControlTargetPositionAndOrientation
  PhysicsCoontrol -> SetControlData(ControlName, ControlData)
```

## Sequencer
- 可添加关卡中的骨骼网格体、Actor、LevelSequencer等
- 可添加当前关卡中的物体
- 可添加Animation
- 可添加Control Rig
- 添加关键帧，按Enter键。  选择一条曲线，比如Transform -> Location -> X, 修改值（可以在关卡中查看变化）


## How to modify animation in UE
如果当前animation不是Mannequin, 先Retarget到Mannequin，然后使用Control Rig Sharing示例的方式做一个Level Sequence, 做Control Rig Sharing。 然后在关卡中通过拖动控制器修改动画，修改完了以后再右键Bake Anim Sequence导出动画。

## Physics Animation
- Below bone physics
```
Event Tick
  SkeletalMeshComponent->SetAllBodiesBelowSimulatePhysics(boneName, bool NewSimulate, bool IncludeSelf)
  SkeletalMeshComponent->SetAllBodiesBelowSimulateBlendWeight(boneName, PhysicsBlendWeight, SkipCustomPhysicsType, bool IncludeSelf)
```
- HitReaction
```
Event Tick
  SkeletalMeshComponent->SetAllBodiesBelowSimulatePhysics(boneName, bool NewSimulate, bool IncludeSelf)
  SkeletalMeshComponent->SetAllBodiesBelowSimulateBlendWeight(boneName, PhysicsBlendWeight, SkipCustomPhysicsType, bool IncludeSelf)

EventHitReaction
  1. if(!isHit) { isHit = true ->TimeLine_0(PlayfromStart) -> Update( Set BlendWeight to NewTrack0) -> Finished(SkeletalMeshComponent->SetAllBodiesPhysicsBlendWeight(PhysicsBlendWeight = 0)->SetAllBodiesSimulatePhysics(false)) -> isHit = false
  2. Delay(0.048) -> Completed( SkeletalMeshComponent->AddImpulse(Impulse, boneName, VelChange = true))
注意isHit的逻辑。

```
- Constraint Profiles
```
Event BeginPlay
  SkeletalMeshComponent->SetAllBodiesBelowSimulatePhysics('spine_01', true, true)

Event CallTriggerActor
  SkeletalMeshComponent->WakeAllRigidBodies()  -> SkeletalMeshComponent->SetConstraintProfileForAll(ProfileName = 'Locked', DefaultIfNotFound = true)

Event CallTriggerActorEndOverlap
  SkeletalMeshComponent->WakeAllRigidBodies()  -> SkeletalMeshComponent->SetConstraintProfileForAll(ProfileName = None, DefaultIfNotFound = true)

```
- Physics Animation Component
```
Event BeginPlay
  PhysicsAnimationComponent->SetSkeletalMeshComponent(component)
  PhysicsAnimationComponent->ApplyPhysicalAnimationSettingsBelow(boneName, Make Physical Animation Data(xxxStrength, xxxStrength, ...), includeSelf = true)
  SkeletalMeshComponent->SetAllBodiesBelowSimulatePhysics(boneName, true, true)
  SkeletalMeshComponent->SetAllBodiesBelowSimulateBlendWeight(boneName, 1, SkipCustomPhysicsType = false, bool IncludeSelf = true)
Event Tick
  PhysicsAnimationComponent->SetStrengthMultiplyer(InStrengthMultiplyer)
```

- SkeletalMeshComponent->Stop() // Stop playing animation
- SkeletalMeshComponent->Play() // Play animation.

## 其他

