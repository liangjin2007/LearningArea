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

智能体(Agent)三个组成部分和类型：
  a.策略Policy: 策略是个函数。确定性策略，随机性策略。 神经网络预测action概率或者直接输出action。
  b.价值函数：
  c.模型：
  d.强化学习智能体的类型： 1. 基于价值与基于策略 ；2. 有模型与免模型。可以先思考在智能体执行动作前，是否能对下一步的状态和奖励进行预测，如果能，就能够对环境进行建模，从而采用有模型学习。（即模型用来模拟真实环境，可预测下一步状态和奖励）
     目前，大部分深度强化学习方法都采用了免模型强化学习

学习与规划:略略略
探索和利用：略略略
强化学习实验: Gym（OpenAI）

马尔可夫决策过程
	马尔可夫过程(MP)：是指一个随机过程在给定现在状态及所有过去状态情况下，其未来状态的条件概率分布仅依赖于当前状态；离散时间的马尔可夫过程称为马尔可夫链；状态转移矩阵。
	马尔可夫奖励过程(MRP)：马尔可夫链+奖励函数。奖励函数是个期望。
		范围horizon：指每个回合最大的时间步数，又称一个回合的长度。
		回报return: Gt = r_(t+1) + gamma * r_(t+2) + gamma^2 * r_(t+3) + ... + gamma^(T-t-1) * r_(T)， 又称为折扣回报。每条轨迹（一个回合的状态序列？）对应一个回报。
		状态价值函数(state-value function): V_t(s) = E(Gt|st=s)， 算期望的通用方法： 蒙特卡洛采样。
		贝尔曼方程计算状态价值函数方法：V(s) = R（s） +  gamma Sum（ p(s'|s) V(s')）。

	计算马尔可夫奖励过程价值的迭代算法（large MRP）：计算状态价值函数V_t(s)
		动态规划法(DP)
		蒙特卡洛方法（MC）： 书中有图，仔细看能看懂，可以理解为有两个维度，时间维度离散化 t {t, t+1，..., H-1}, 状态维度离散化为一个轨迹 {s_t, s_(t+1), ...s_(H-1)}
		时序差分学习（TD Learning）：前两种方法的结合



	
```
- Policy Function和Value Function
![PolicyAndValue](https://github.com/liangjin2007/data_liangjin/blob/master/PolicyFunctionAndValueFunction.jpg?raw=true)








## UE5 Learning Agents https://dev.epicgames.com/community/learning/courses/GAR/unreal-engine-learning-agents-5-5/7dmy/unreal-engine-learning-to-drive-5-5

### 概述
- Learning Agents is a plugin that allows game developers to train and deploy bots in Unreal Engine using Reinforcement Learning and Imitation Learning.
```
Learning Agents comes with both a C++ library (with Blueprint support) and Python scripts. The C++ library is an ordinary UE plugin split into a handful of modules. It exposes functionality for defining observations/actions and your neural network structure, as well as the flow-control for the training and inference procedures. During training, the UE process will collaborate with an external Python process running PyTorch. We included a working PyTorch algorithm for PPO and BC.
```

- 最新 蓝图 API https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/LearningAgents
- 最新 C++ API https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Plugins/LearningAgents

- Notes: 中间有重构，所以需要考虑使用哪个版本，否则代码差异解决不了。
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


- 一个强化学习的基本模块  
```
An Interactor (Observe Data and Apply Actions)
A Trainer (Give rewards And Execute Completions)
A Policy (Decides What the agent does with the given Observations)
Previously, these three are all ULearningAgentsManagerComponent type components, so would be added to an actor
```

```
1. ULearningAgentsManager now is a UActorComponent, so need to add to an actor
2. Previous ULearningAgentsManagerComponent type object all becomes ULearningAgentsManagerListener object which is a UObject
	ULearningAgentsImitationTrainer
	ULearningAgentsPPOTrainer
	ULearningAgentsRecorder
	ULearningAgentsTrainingEnvironment
	ULearningAgentsController
	ULearningAgentsCritic
	ULearningAgentsInteractor
	ULearningAgentsPolicy
3. All the input/output is now flatted to a vector.
4. ULearningAgentsPolicy now has the following three objects.
	EncoderObject  // TSharedPtr<UE::Learning::FNeuralNetworkFunction>
	DecoderObject  // TSharedPtr<UE::Learning::FNeuralNetworkFunction>
	PolicyObject   // TSharedPtr<UE::Learning::FNeuralNetworkPolicy>
	EncoderNetwork // TObjectPtr<ULearningAgentsNeuralNetwork>
	PolicyNetwork  // TObjectPtr<ULearningAgentsNeuralNetwork>
	DecoderNetwork // TObjectPtr<ULearningAgentsNeuralNetwork>

void ULearningAgentsPolicy::RunInference(const float ActionNoiseScale)
{
	Interactor->GatherObservations();
	Interactor->MakeActionModifiers();
	EncodeObservations();
	EvaluatePolicy();
	DecodeAndSampleActions(ActionNoiseScale);
	Interactor->PerformActions();
}

ULearningAgentsPolicy::EvaluatePolicy()
		PolicyObject->Evaluate(
		ActionVectorsEncoded,
		MemoryState,
		ObservationVectorsEncoded,
		PreEvaluationMemoryState,
		ValidAgentSet);


ULearningNeuralNetworkData: UObject
	private:
		// Uploads the FileData to NNE and updates the internal in-memory representation of the network used for inference.
		void UpdateNetwork(); // Update ModelData and Network by FileData.

		UPROPERTY()
		TArray<uint8> FileData;
		
		// Model data used by NNE
		UPROPERTY()
		TObjectPtr<UNNEModelData> ModelData;

		// Internal in-memory network representation
		TSharedPtr<UE::Learning::FNeuralNetwork> Network;


ModleData -> UE::NNE::IModelCPU对象 -> Network's UE::Learning::FNeuralNetworkInference typed array TArray<TWeakPtr<FNeuralNetworkInference>, TInlineAllocator<64>> InferenceObjects
	TWeakInterfacePtr<INNERuntimeCPU> RuntimeCPU = UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeBasicCpu"));
	
	TSharedPtr<UE::NNE::IModelCPU> UpdatedModel = nullptr;
	
	if (ensureMsgf(RuntimeCPU.IsValid(), TEXT("Could not find requested NNE Runtime")))
	{
		UpdatedModel = RuntimeCPU->CreateModelCPU(ModelData);
	}
	Network->UpdateModel(UpdatedModel, InputSize, OutputSize); // 

UE::Learning::FNeuralNetworkInference::Evaluate(TLearningArrayView<2, float> Output, const TLearningArrayView<2, const float> Input)

```





- UE5.3 推理部分请搜索::RunInference(
```
ULearningAgentsPolicy::RunInference()
	Interactor->EncodeObservations();
	EvaluatePolicy();
	Interactor->DecodeActions();

具体地：
ULearningAgentsInteractor::EncodeObservations(const UE::Learning::FIndexSet AgentSet)
  ...
  Observations->Encode(ValidAgentSet); // jump to FConcatenateFeature::Encode(const FIndexSet Instances)
  ...

  ISPC代码优化性能，通常比普通c++代码有数倍的性能提升。目前使用宏UE_LEARNING_ISPC默认开启。

ULearningAgentsPolicy::EvaluatePolicy()
  ...
  PolicyObject->Evaluate(ValidAgentSet); // TSharedPtr<UE::Learning::FNeuralNetworkPolicyFunction> PolicyObject;
  ...

->
FNeuralNetworkPolicyFunction::Evaluate(const FIndexSet Instances)
  // 第一步、先从InstanceData获取Inputs（二维数组）, Outputs（二维数组）, OutputMeans（二维数组）,  OutputStds（二维数组），Seed（一维数组）, ActionNoiseScale（一维数组）。所以InstanceData应该是个非常重要的数据对象。
  // NeuralNetwork->GetInputNum() == Inputs.Num<1>()          // TSharedRef<FNeuralNetwork> NeuralNetwork; 
  // NeuralNetwork->GetOutputNum() == 2 * Outputs.Num<1>()     
```

- UE5.6 推理部分
```
ISPC相关代码已经移除，增加了对NNE的依赖
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"




```

- 用UE5.6 API写个推理的例子。 正好用Metamotivo作为示例。




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

## MuJoCo Documentation 
- MJCF https://mujoco.readthedocs.io/en/stable/XMLreference.html
- Documentation https://mujoco.readthedocs.io/en/stable/overview.html

## npy to bvh 
- AMASS https://github.com/facebookresearch/humenv/tree/main/data_preparation
- https://github.com/KosukeFukazawa/CharacterAnimationTools#13-load-animation-from-amass
- https://github.com/KosukeFukazawa/smpl2bvh?tab=readme-ov-file
- AMASS https://amass.is.tue.mpg.de/download.php
- SMPL https://smpl.is.tue.mpg.de/download.php
- SMPL-H https://mano.is.tue.mpg.de/, to process AMASS, download Extended SMPL+H model
- SMPL-X https://github.com/vchoutas/smplx

## humenv
- https://github.com/facebookresearch/humenv/tree/main/data_preparation

## Genesis
- 中文文档 https://genesis-world.readthedocs.io/zh-cn/latest/user_guide/overview/what_is_genesis.html

## Boss 战 https://www.bilibili.com/video/BV1BD421H7xt/?spm_id_from=333.337.search-card.all.click&vd_source=8cfcd5a041c19f2f5777e1a8d78359f2
  
