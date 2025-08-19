# 参考链接 
- https://zhuanlan.zhihu.com/p/676940299

## 概念
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

智能体(Agent)组成部分：
  策略
```

- 强化学习目标
![强化学习目标](https://github.com/liangjin2007/data_liangjin/blob/master/rl.jpg?raw=true)

- 基于价值 (Value-Based)和基于策略(Policy-Based)的区别
- 贝尔曼公式

## UE5 Learning Agents https://dev.epicgames.com/community/learning/courses/GAR/unreal-engine-learning-agents-5-5/7dmy/unreal-engine-learning-to-drive-5-5

### 概述
Learning Agents comes with both a C++ library (with Blueprint support) and Python scripts. The C++ library is an ordinary UE plugin split into a handful of modules. It exposes functionality for defining observations/actions and your neural network structure, as well as the flow-control for the training and inference procedures. During training, the UE process will collaborate with an external Python process running PyTorch. We included a working PyTorch algorithm for PPO and BC.




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

代码2 https://github.com/XanderBert/Unreal-Engine-Learning-Agents-Learning-Environment
```




