# 创建交互体验

- [1.人工智能](#1人工智能)
- [2.Gameplay相机系统](#2Gameplay相机系统)
- [3.Gameplay定位系统](#3Gameplay定位系统)
- [4.输入](#4输入)
- 
## 1.人工智能 
https://dev.epicgames.com/documentation/zh-cn/unreal-engine/artificial-intelligence-in-unreal-engine
- 行为树
  - 用于为非玩家角色创建AI
- MassEntity Experimental
  - 用于实现群体避障 
- 寻路系统
  - 虚幻引擎寻路系统 允许人工智能代理通过寻路功能在关卡中走动。

- 神经网络引擎
  - https://github.com/Akiya-Research-Institute/NNEngine-Demo
  - https://github.com/Akiya-Research-Institute/Artistic-Style-Transfer-on-UE4
  - https://github.com/Akiya-Research-Institute/Monocular-Depth-Estimation-on-UE
- 智能对象
- StateTree  
- 场景查询系统EQS Experimental功能 https://dev.epicgames.com/documentation/zh-cn/unreal-engine/environment-query-system-in-unreal-engine
- AI感知
- AI调试
- AI组件 AI Perception Component

## 2.Gameplay相机系统 
Experimental
后面可以看看是否可以做这个事情。 剧情/角色切换镜头。

## 3.Gameplay定位系统 
Targeting System  Beta
插件 Targeting System

## 4. 输入
https://dev.epicgames.com/documentation/zh-cn/unreal-engine/input-overview-in-unreal-engine
- ActionMapping 将离散按钮或按键映射到一个"友好的名称"，该名称稍后将与事件驱动型行为绑定
- AxisMapping for 连续行为，会做轮询
- 项目设置 -> 输入 -> Action Mapping
- 项目设置 -> 输入 -> Axis Mapping
- InputComponent这个跟C++中Component的概念不同，指的是Actor, PlayerController, Level Blueprint, Pawn都可以处理Input, 所以有优先级顺序。
![InputComponent](https://d1iv7db44yhgxn.cloudfront.net/documentation/images/f5592819-a96b-4384-b9fe-b38fe5494942/inputflow.png)


Gameplay框架 https://dev.epicgames.com/documentation/zh-cn/unreal-engine/gameplay-framework-in-unreal-engine?application_version=5.4
Gameplay技能系统 https://dev.epicgames.com/documentation/zh-cn/unreal-engine/gameplay-ability-system-for-unreal-engine?application_version=5.4
物理 https://dev.epicgames.com/documentation/zh-cn/unreal-engine/physics-in-unreal-engine?application_version=5.4
碰撞 https://dev.epicgames.com/documentation/zh-cn/unreal-engine/collision-in-unreal-engine?application_version=5.4
