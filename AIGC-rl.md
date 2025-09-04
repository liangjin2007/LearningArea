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
绪论：
	智能体（Agent）：学习和决策的主体。
	环境（Environment）：智能体交互的外部世界。
	状态（State）：环境在某一时刻的描述。
	动作（Action/Decision）：智能体在某一状态下可以执行的操作。
	奖励（Reward）：智能体执行动作后从环境中获得的反馈。
	策略（Policy）：智能体在给定状态下选择动作的规则。(策略分为确定性策略(deterministic policy)和不确定性策略(stochastic policy),确定性策略只每种状态下选择的action是确定的即百分之百选择一种策略，而不确定性策略指某种状态下选择的action是不确定的，策略只是确定action的分布，          然后在分布中进行采样来确定策略)
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

第一章、马尔可夫决策过程
	马尔可夫过程(MP)：
		是指一个随机过程在给定现在状态及所有过去状态情况下，其未来状态的条件概率分布仅依赖于当前状态；离散时间的马尔可夫过程称为马尔可夫链；状态转移矩阵。
		p(s_(t+1)|st=s)

	马尔可夫奖励过程(MRP)：马尔可夫链+奖励函数。奖励函数是个期望
		范围horizon：指每个回合最大的时间步数，又称一个回合的长度。
        奖励序列: r_(t+1), r_(t+2), ...
		回报return: Gt = r_(t+1) + gamma * r_(t+2) + gamma^2 * r_(t+3) + ... + gamma^(T-t-1) * r_(T)， 又称为折扣回报。每条轨迹（一个回合的状态序列？）对应一个回报。
		状态价值函数(state-value function): V_t(s) = E(Gt|st=s)， 算期望的通用方法： 蒙特卡洛采样。
		贝尔曼方程计算状态价值函数方法：V(s) = R（s） +  gamma Sum（ p(s'|s) V(s')）。
        奖励函数R(st=s) = E[r_(t+1)|st=s]

	计算马尔可夫奖励过程价值的迭代算法（large MRP）：计算状态价值函数V_t(s)
		动态规划法(DP): bootstrapping的方法不停地迭代贝尔曼方程，书中有图
		蒙特卡洛方法（MC）： 书中有图，仔细看能看懂，可以理解为有两个维度，时间维度离散化 t {t, t+1，..., H-1}, 状态维度离散化为一个轨迹 {s_t, s_(t+1), ...s_(H-1)}
		时序差分学习（TD Learning）：前两种方法的结合

	马尔可夫决策过程：指多了决策（action）
		状态转移: p(s_(t+1)|st = s, at = a)
		奖励序列：rt(st=s, at=a)
        奖励函数: 变成R(st=s, at = a) = E[rt|st=s, at=a]
        策略Policy: pi(a|s) = p(at=a|st=s)
        已知马尔可夫决策过程和策略pi， 可以将马尔可夫决策过程转换成马尔可夫奖励过程。 p(s'|s) = sum pi(a|s)p(s'|s, a)
		马尔可夫决策过程的状态转移对比： 书上有图比较形象
		状态价值函数：V_pi(s) = E_pi(Gt|st=s),也就是期望基于采取的策略
		Q函数(Q-function)或者叫动作价值函数。只的是在某一个状态采取某一个动作，它有可能得到的回报的期望  Q_pi(s, a) = E_pi[G_t|st=s, at=a] = R(s, a) + gamma sum_s' p(s'|s, a) V_pi(s'), 这里的期望其实也是基于策略的。
        状态价值函数：前面的V_pi(s) = E_pi[Gt|st=s] = sum pi(a|s) Q_pi(s, a)
		奖励函数R_pi(st=s, at=a) = E[r_(t+1)|st=s, at=a]

		备份图(backup)：跟如何做策略评估等有关，建议好好读一读。
			书上有图： 图2.10画错了。
			状态价值函数分解计算，分解为两步：先将未来状态上价值，先算出 状态-动作对 上的价值Q_pi(s, a)， 再算状态s上的价值V_pi(s), 也就是下面：
				先算 Q_pi(s, a) = R(s, a) + gamma sum_s' p(s'|s, a) V_pi(s')
				再算 V_pi(s) = sum_a pi(a|s) Q_pi(s, a)
			动作价值函数分解计算，同理也分为两步：
				先算 V_pi(s') = sum_a' pi(a'|s') Q_pi(s', a')
				再算 Q_pi(s, a) = R(s, a) + gamma sum_s' p(s'|s, a) V_pi(s')


		策略评估/价值预测：已知马尔可夫决策过程以及要采取的策略pi，计算（状态）价值函数V_pi(s)的过程就是策略评估。即下面的预测。

		预测(prediction)：输入<S, A, P, R, gamma>和策略pi， 输出是价值函数V_pi(s), 也就是计算每个状态的价值。
		控制(control)： 搜索最佳策略。 输入<S, A, P, R, gamma>， 输出最佳价值函数V*和最佳策略pi*。 也就是去找一个最佳的策略，同时输出它的最佳价值函数。
			这两者是递进关系，在强化学习中，通过解决预测问题，进而解决控制问题。

		动态规划

		马尔可夫决策过程中的策略评估：
			方法： 把贝尔曼期望备份转换成动态规划的迭代   当得到上一时刻的V^t的时候，去得到下一时刻的V^(t+1)(s)
			V^(t+1)(s) = sum_a pi(a|s) (R(s, a) + gamma sum_s' p(s'|s, a) V^t(s'))
			把上式的贝尔曼期望备份反复迭代，然后得到一个收敛的价值函数的值。因外pi(a|s)已知，所以上式可简化成一个马尔可夫奖励过程的表达形式 V_(t+1)(s) = r_pi(s) + gamma P_pi(s'|s) V_t(s')

		马尔可夫决策过程控制
			Step 1. V*(s) = max_pi V_pi(s)， 即优化一个策略网络pi(s)，使得每个状态的价值最大。最佳策略 pi*(s) = arg max_pi V_pi(s), 此网络输入只有s。此时每个V(s)都是最大值。
			Step 2. pi*(a|s) = {1, a = arg max_a Q*(s, a), 对Q函数最大化; 0, 其他
			Q:怎么进行策略搜索? A: 1.群举； 2.其他方法

		策略迭代：
			由两个步骤组成：策略评估和策略改进。策略是个函数，就可以做优化。
			for i in 迭代次数
				固定策略，算状态价值函数，进一步可以算出Q函数（状态动作价值函数）, 对Q函数进行最大化（在Q函数做贪心搜索来改进策略（函数））
				重复。
			Q表格（Q-table）
			贝尔曼最优方程： 上述迭代停止后，得到V_pi(s) = max_a Q_pi(s, a)

		价值迭代：
			最优星原理定理（principle of optimality theorem）：一个策略pi(a|s)在状态s达到了最优价值，也就是V_pi(s) = V*(s)成立，当且仅当对于任何从s到达s',都已经达到了最有价值，也就是对于所有的s', V_pi(s') = V*(s')。
			确认行价值迭代
			价值迭代算法：
				(1)初始化： 令k = 1, 对于所有状态s， V0(s) = 0
				(2)对于k = 1:H（H是让V(s)收敛所需的迭代次数）
					(a) 对于所有状态s
						Q_(k+1)(s, a) = R(s, a) + gamma sum_s' p(s'|s, a) V_k(s')
						V_(k+1)(s) = max_a Q_(k+1)(s, a)
					(b) k <—— k+1
				(3)在迭代后提取最优策略：
					pi(s) = arg max_a [R(s, a) + gamma sum_s' p(s'|s, a) V_(H+1)(s')]

		策略迭代与价值迭代的区别

第二章、表格型方法
	略

第三章、策略梯度
	智能体、一个策略和一个演员
 	略
    ∇ ¯R_θ = E_(τ∼p_θ(τ)) [R(τ )∇ log p_θ(τ )]

第四章、PPO 近端策略优化
	重要性采样
		同策略与异策略
		integrate(f(x) p(x))dx = integrate(f(x) p(x)/q(x) q(x)dx = E_(x~q)[f(x) p(x)/q(x)]
		重要性权重 importance weight p(x)/q(x)
		Var[X] = E [X^2] − (E[X])^2
	KL散度
	
第五章、深度Q网络




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
- AMASS npy to bvh https://github.com/KosukeFukazawa/CharacterAnimationTools#13-load-animation-from-amass
- https://github.com/KosukeFukazawa/smpl2bvh?tab=readme-ov-file
- AMASS https://amass.is.tue.mpg.de/download.php    用户名liangjin2007@gmail.com
- SMPL https://smpl.is.tue.mpg.de/download.php      用户名jl5400@163.com
- SMPL-H https://mano.is.tue.mpg.de/, to process AMASS, download Extended SMPL+H model   用户名jl5400@163.com
- SMPL-X https://github.com/vchoutas/smplx
- SMPL-X, SMPL-H, etc blender add on https://github.com/Meshcapade/SMPL_blender_addon   缺data目录，用不了。最后使用smplx网站上提供的smpl_blender_addon插件，导出smplx neutral mesh，导出时去掉poses correctives，及导出前先snap to ground和设置texture为female。
- Textured SMPL https://github.com/Meshcapade/SMPL_texture_samples   没用这个，因为前面的导出smplx neutral mesh已经可以导出texture到fbx里面。
- SMPL-x https://smpl-x.is.tue.mpg.de/  用户名jl5400@163.com
  
## humenv
- https://github.com/facebookresearch/humenv/tree/main/data_preparation
- 尝试转化amass数据集以使得能生成部分调试数据
```
创建humenv环境
conda create -n humenv python=3.10

安装SMPLSim
下载https://github.com/ZhengyiLuo/SMPLSim
cd SMPLSim-master
conda activate humenv
pip install -r .\requirements.txt

发现https://github.com/ZhengyiLuo/smplx装不上。
先下载下来
cd smplx-master
pip install .

继续安装SMPLSim
cd ../SMPLSim-master
在requirements.txt中注释掉smplx那一行（前面加#）。
pip install .

尝试安装PHC
从https://github.com/ZhengyiLuo/PHC下载PHC
放到SMPLSim-master同级目录
将https://github.com/ZhengyiLuo/PHC/blob/master/scripts/data_process/process_amass_db.py文件拷贝到humenv-master/data_preparation里

配置vscode
用vscode打开humenv-master代码
Ctrl+Shift+P，点击Python Interpretor, 选择humenv那个python。
打开data_preparation/process_amass.py
安装h5py
安装rich
pip install rich h5py

```

## SMPL
- build qpos https://github.com/ZhengyiLuo/SMPLSim/blob/master/examples/motion_test.py


## Genesis
- 中文文档 https://genesis-world.readthedocs.io/zh-cn/latest/user_guide/overview/what_is_genesis.html

## PHC and SMPLSim
- Perpetual Humanoid Control for Real-time Simulated Avatars https://github.com/ZhengyiLuo/PHC
- SMPLSim: Simulating SMPL/SMPLX Humanoids in MUJOCO and Isaac Gym https://github.com/ZhengyiLuo/SMPLSim

## Boss 战 https://www.bilibili.com/video/BV1BD421H7xt/?spm_id_from=333.337.search-card.all.click&vd_source=8cfcd5a041c19f2f5777e1a8d78359f2
  
