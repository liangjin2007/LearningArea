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
动作（Action）：智能体在某一状态下可以执行的操作。
奖励（Reward）：智能体执行动作后从环境中获得的反馈。
策略（Policy）：智能体在给定状态下选择动作的规则。(策略分为确定性策略(deterministic policy)和不确定性策略(stochastic policy),确定性策略只每种状态下选择的action是确定的即百分之百选择一种策略，而不确定性策略指某种状态下选择的action是不确定的，策略只是确定action的分布，然后在分布中进行采样来确定策略)
价值函数（Value Function）：评估在某一状态下长期累积奖励的期望值。
```

- 强化学习目标
![强化学习目标](https://github.com/liangjin2007/data_liangjin/blob/master/rl.jpg?raw=true)

- 基于价值 (Value-Based)和基于策略(Policy-Based)的区别
- 贝尔曼公式

## UE5 Learning Agents https://dev.epicgames.com/community/learning/courses/GAR/unreal-engine-learning-agents-5-5/7dmy/unreal-engine-learning-to-drive-5-5
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
