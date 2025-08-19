# 参考链接 
- https://zhuanlan.zhihu.com/p/676940299
- EasyRL_v1.0.6.pdf

## 强化学习概念
```
强化学习（Reinforcement Learning, RL）是机器学习的一个重要分支，主要研究智能体（Agent）如何在一个环境中通过与环境交互来学习策略，以最大化某种累积奖励。
强化学习的核心思想是通过试错（Trial and Error）来学习，智能体通过执行动作（Action）来影响环境，并从环境中获得反馈（Reward），进而调整其策略（Policy）以优化长期奖励。
```

- 强化学习的基本要素
```
智能体（Agent）：学习和决策的主体。
环境（Environment）：智能体交互的外部世界。
状态（State）：环境在某一时刻的描述。
动作（Action/Decision）：智能体在某一状态下可以执行的操作。
奖励（Reward）：智能体执行动作后从环境中获得的反馈。
策略（Policy）：智能体在给定状态下选择动作的规则。(策略分为确定性策略(deterministic policy)和不确定性策略(stochastic policy),确定性策略只每种状态下选择的action是确定的即百分之百选择一种策略，而不确定性策略指某种状态下选择的action是不确定的，策略只是确定action的分布，然后在分布中进行采样来确定策略)
价值函数（Value Function）：评估在某一状态下长期累积奖励的期望值。
预演（Rollout）及最终奖励（Eventual Reward）：预演是指我们从当前帧对动作进行采样，生成很多局游戏。
轨迹（Trajectory）：(s0, a0, s1, a1, ...)
回合Episode/试验Trial
期望累计奖励（Expected Cummulative reward）
序列决策（sequential decision making）
历史（Ht）: Ht = o1，a1, r1, o2, a2, r2, ..., ot, at, rt
观测（Obervation）与状态（State）的关系： 状态是对世界的完整描述，不会隐藏世界的信息，观测是对状态的部分描述。
完全可观测的（fully observed） ——>  用马尔可夫决策过程MDP建模。
部分可观测的（Partially observed）——> 用POMDP建模。
动作空间（Action Space）: 离散动作空间，连续动作空间

智能体(Agent)三个组成部分：
  策略Policy: 策略是个函数。确定性策略，随机性策略。 神经网络预测action概率或者直接输出action。
  价值函数：
  模型：


```
- Policy Function和Value Function
![PolicyAndValue](https://github.com/liangjin2007/data_liangjin/blob/master/PolicyFunctionAndValueFunction.jpg?raw=true)








## UE5 Learning Agents https://dev.epicgames.com/community/learning/courses/GAR/unreal-engine-learning-agents-5-5/7dmy/unreal-engine-learning-to-drive-5-5

### 概述
```
Learning Agents comes with both a C++ library (with Blueprint support) and Python scripts. The C++ library is an ordinary UE plugin split into a handful of modules. It exposes functionality for defining observations/actions and your neural network structure, as well as the flow-control for the training and inference procedures. During training, the UE process will collaborate with an external Python process running PyTorch. We included a working PyTorch algorithm for PPO and BC.
```
- 蓝图API https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/LearningAgents

Learning Agents is a plugin that allows game developers to train and deploy bots in Unreal Engine using Reinforcement Learning and Imitation Learning.

```
Changes from 5.4 to 5.5

Re-Architect ML Training Code

Rewrite the ML training code such that networking protocols and ML training algorithms are able to be independently developed

Protocols need to accept a variable number of networks & data buffers

Support Additional Networking Protocols

TCP Sockets

Shared Memory

Bring Your Own Training Algorithm

Rewrite the python side of Learning Agents so that it is much easier for users to integrate off the shelf ML training algorithms or custom python training code

Make sure these changes are easy to make without users needed to hack up code which will be touched by us, i.e. no merge conflicts

Make necessary changes to the C++ side so that users can select their python training code and pass additional training algorithm arguments


Mac and Linux Training Added
```
  


```
Step 1. 创建一个带LearningAgentsManager Componenet的Actor Named with BP_SportsCarManager
Click Class Default to show Detail, add Component Tag "LearningAgentsManager"

Step 2. For each of the Car Pawn, in the Begin Play event, Find the LearningAgentsManager Actor, get the component, call AddAgent to add the pawn object.
There is Logging informatin for AddAgent function.

Step 3. Manager Listeners : LearningAgentsManagerListener
三个派生类：
  LearningAgentsInteractor： 定义Agents如何与世界交互（通过Observation和Action）
    需要重载以下四个函数：
      SpecifyAgentObservation
      SpecifyAgentAction
      GatherAgentObservation
```


### 示例代码
```
代码1 https://github.com/automathan/ue-la-example?tab=readme-ov-file
及其中的文档
文档1 https://medium.com/@gensen/early-explorations-of-learning-agents-in-unreal-engine-ef74b058161e
文档2 https://medium.com/@gensen/killers-and-explorers-training-rl-agents-in-unreal-engine-7976a83b01d7

安装UE5.3.2

ue-la-example 为UE5.3版本示例。为第三方作者写的程序，缺少一个Config文件，暂时跑不起来。
LearningAgents的API在5.3到5.6（具体是5.4.4到5.5.0, 具体可以看前面5.4升到5.5，Epic对该插件做了重构）过程中API发生了较大变化，最新的API官方并无发布任何示例程序。

调试技巧： 拷贝UE源码中的LearningAgents插件源代码到ue-la-example，将LearningAgents.uplugin重命名为LearningAgentsV2.uplugin, 修改.uproject中的插件名为LearningAgentsV2。这样可以把LearningAgents的源码加到游戏项目中，方便设断点调试。

```

- **看这个**
```
代码2 https://github.com/XanderBert/Unreal-Engine-Learning-Agents-Learning-Environment
拷贝ue-la-example中的LearningAgents目录到此项目，将引擎依赖为UE5.3。
此github链接中有文档
```
- 如何将神经网络输出注入到行为树去控制行为树
- 

## UE行为树
- 开源代码 https://github.com/EugenyN/UE5_Demo1?tab=readme-ov-file

## 如何利用UE的物理引擎自己写个专门的物理模拟功能。
- 参考UNetworkPhysicsComponent
- 在引擎代码中搜xxxPhysicsComponent： 实现一个功能基本上会用Component
  
