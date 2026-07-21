- musclemimic https://github.com/amathislab/musclemimic

- “真实地复现人体动力学模型”的3D setup在科学研究、仿生学和计算机图形学领域早已发展成熟。根据你的侧重点（生物肌骨分析、力学碰撞还是人工智能仿真），目前主流且最顶尖的系统和平台如下：1. 专业的生物力学与肌骨系统仿真 (Musculoskeletal Simulation)这类 Setup 结合了骨骼解剖学与肌肉生理学，能够精确计算肌肉力量、关节负荷和肌腱张力。AnyBody Modeling System：全球顶级的商业级人体肌骨动力学分析平台。它通过输入运动学数据，利用逆动力学计算出最真实的内部组织负荷，被广泛应用于康复工程、汽车人体工效学和外骨骼设计。OpenSim：由斯坦福大学主导开发的顶级开源生物力学平台。它允许用户通过数学和物理模型自由开发复杂的肌肉驱动仿真。配合 OpenCap，你只需用两部智能手机拍摄视频，即可在云端几分钟内快速建立3D动力学和肌肉力分析 Setup。For an overview of how these tools use motion data to solve complex biological joint and muscle dynamics, watch this demonstration:44sOpenSim: A Powerful Tool for Muscle-Driven SimulationBiomechanics of MovementYouTube · Jul 30, 20222. 多体动力学与通用工程碰撞仿真 (Multibody Dynamics)Ansys Motion：内建强大的生物力学模块，能够进行人体多柔体系统（多刚体加肌肉建模）的耦合仿真。它能完美复现人体与外界（如汽车座椅、防护设备或机械）发生强烈接触或碰撞时的动力学响应。SOLIDWORKS Simulation Premium：虽然主要针对机械，但也内含事件基础的动态运动（Motion）仿真，可计算关节和连杆系统的瞬态动力学。3. AI与机器人学中的高保真世界仿真 (Robotics & World Models)在具身智能（Embodied AI）和仿生机器人领域，研究人员常常将高保真人偶嵌入物理引擎，用于人类行为和控制策略的模拟。NVIDIA Isaac Lab：基于 GPU 加速的机器人学习与仿真框架，支持全身控制，适合大规模多模态的数据合成和人体全身运动动力学强化学习。Cosmos 3：NVIDIA推出的全模态世界模型，统一理解视觉与动作序列，提供极高物理一致性的世界模拟环境。MuJoCo / Gym：作为 AI 强化学习最常用的轻量级高频物理引擎，大量用于基于 3D 骨骼和关节约束（如通用的 SMPL 人体模型）的物理动态对齐研究

```
机器人与人体力学交互（Physical Human-Robot Interaction, 简称 pHRI 或 共融机器人）是当前具身智能与医疗康复领域的核心交叉学科。重点研究包括：阻抗/导纳控制（让机器人像人一样柔顺）、人体生物力学建模（理解肌肉骨骼受力） 及 人机意图识别（预测动作）。以下是该领域的核心会议、代表性论文及顶级期刊指南：一、 顶级学术会议在这些会议中，人体力学与机器人交互（Bio-robotics & pHRI）通常是核心议题：IEEE/RSJ IROS (International Conference on Intelligent Robots and Systems)特点：机器人领域规模最大的顶级综合会议之一。大量关于人机协作碰撞安全、柔顺控制与外骨骼的论文在此发表。IEEE ICRA (International Conference on Robotics and Automation)特点：机器人领域的风向标。包含专门的“Human-Robot Interaction”分会场，涵盖动力学、步态分析和仿生机械学。IEEE RAS BioRob (International Conference on Biomedical Robotics and Biomechatronics)特点：最对口的专业会议。完全聚焦生物医学机器人、人机物理交互、神经力学和假肢技术。ACM/IEEE HRI (International Conference on Human-Robot Interaction)特点：人机交互界的顶级会议。更偏向用户研究、多模态意图理解与自然语言控制交互。IEEE-RAS Humanoids (IEEE International Conference on Humanoid Robots)特点：人形机器人顶级盛会，高度关注仿生力学、拟人运动控制与人体肌肉骨骼建模。二、 经典与代表性论文推荐理解该领域，建议从以下几篇被广泛引用的综述与经典力学控制论文入手：人机协作控制综述：How Robots and People Communicate Through Physical Interaction (2023) - 发表于 Annual Review of Control, Robotics, and Autonomous Systems。深入梳理了从导纳控制到基于推理的高级意图交流算法。Physical human-robot interaction: A critical review of safety, control, and architecture - 行业经典综述，详述了协作机器人（Cobot）的安全设计、本质柔顺关节（如串联弹性驱动器 SEA）以及力控方案。生物力学与交互控制：Learning Whole-Body Human-Robot Haptic Interaction in Social Contexts (ICRA) - 探讨全身触觉交互，如何通过从人类示范中学习（LfD），实现机器人与人体理学交互。Intent aware adaptive admittance control for physical Human-Robot Interaction (IEEE) - 经典文献，探讨了如何在物理交互中实时自适应调节机器人的阻抗模型，以匹配不同人体的力量和技能水平。前沿动态：Bidirectional Human-Robot Communication for Physical Human-Robot Interaction - 近期 HRI 顶级会议成果，展示了如何用大语言模型辅助人类实时修改机器人的力学轨迹和力矩。Real-Time Muscle Biomechanical Modeling with Physics-Informed Neural Networks - 探讨如何结合物理信息神经网络（PINN）与人体肌肉力学，实现软组织形变和受力实时预测。三、 顶级学术期刊追踪如果要深入跟踪最新研究，建议直接锁定以下期刊：IEEE Transactions on Robotics (T-RO)（机器人领域最权威期刊，包含大量精密的pHRI动力学控制论文）。Science Robotics（机器人顶刊，侧重跨学科突破如软体机器人、仿生肌肉与外骨骼）。IEEE Transactions on Neural Systems and Rehabilitation Engineering（生物医学康复工程，侧重下肢外骨骼、假肢与人体生物力学的交互）。Robotics and Computer-Integrated Manufacturing（工业级的人机物理协同）。如果您正在寻找特定方向的文献，请告诉我：您的应用场景是医疗康复外骨骼、工业协作机器人（Cobot）还是人形机器人？您更侧重于底层力控算法（如阻抗控制）还是人体生理/生物力学建模？
```

```
Researchers and engineering labs have successfully created musculoskeletal simulators using MuJoCo, and the recent evolution of GPU-accelerated solvers like MJWarp (powered by the Newton physics core) makes large-scale reinforcement learning for muscle-actuated systems highly scalable.Key Platforms and Implementations1. MuJoCo and MJWarp (DeepMind / NVIDIA Newton)Who did it: The AMathis Lab built MuscleMimic, a JAX-based motion imitation learning benchmark that specifically targets full-body, muscle-actuated models using retargeted motion datasets and MuJoCo/MJWarp dynamics.Why it works: MuJoCo has native built-in support for muscle and tendon actuators in its XML format. The recent integration of Newton / MJWarp (co-developed by DeepMind, NVIDIA, and Disney) allows developers to run massively parallel, GPU-accelerated muscle training.Resources: You can explore the MuscleMimic GitHub Repository and the MuJoCo Warp Documentation for implementation details.2. Isaac Lab (via Newton Physics)Who did it: Teams at ETH Zurich (Robotic Systems Lab) and the Technical University of Munich are actively leveraging NVIDIA's Newton engine, which utilizes MuJoCo-Warp solvers inside Isaac Lab.Why it works: Because Newton merges MuJoCo's advanced biomechanical/tendon actuator capabilities with Isaac Lab's PyTorch-based RL environment, it serves as an ideal framework for training physically accurate musculoskeletal agents.Resources: Learn more about the Newton integration via the Isaac Lab Newton Documentation.3. Legacy: Isaac Gym & PyBulletWho did it: In the past, researchers at universities built custom force-application layers on top of Isaac Gym to approximate muscle actuation, and various labs utilized PyBullet with OpenSim for bio-mechanical walking controllers.Why Choose MuJoCo / NewtonWhile engines like Genesis are popular for rigid-body RL, MuJoCo remains the gold standard for muscle simulation. It provides a robust, physically accurate continuous-time formulation for tendons, slider-crank actuators, and dry friction required by biomechanical models.For tips on how to properly model the mechanics of muscle fibers and tendons within a physics engine:58s6. Muscle Sim Properties Overview2.3K views · 1 year agoYouTube · HoudiniCould you tell me a bit more about your specific project?Are you aiming to model human locomotion/gait, or upper limb dexterous manipulation?Are you planning to use reinforcement learning (simulating thousands of parallel agents) or focus more on biophysical analysis (physics of a single subject)?I can help you narrow down the best tool stack and specific APIs for your goals.
```




- Chaos Flesh
```
https://dev.epicgames.com/community/learning/tutorials/W4mV/unreal-engine-chaos-flesh-muscle-simulation-tutorial-5-5
```

- Ziva
```
https://github.com/Z-Anatomy/Models-of-human-anatomy
```

- Robot-Interaction
```
[2026]Embodied Human Simulation for Quantitative Design and Analysis of Interactive Robotics
```

- opensim-core
```
https://github.com/opensim-org/opensim-core, 1.1k star
12年前开始有初始提交，像是某个实验室的代码。

```

- Survey
```
[2026]3D Generation for Embodied AI and Robotic Simulation: A Survey
```

- 人体解剖学器官结构
```
https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html
https://lifesciencedb.jp/bp3d/
```


- MASS Pytorch based 人体肌肉模拟举重 https://github.com/lsw9021/MASS


- ProtoMotions3
```
https://www.linkedin.com/posts/mahmoudrabie2004_opensourceaiprojects-didyouknowthat-activity-7402446984105689088-K53x
```

- assistive-gym
```
https://github.com/FedericoPivotto/assistive-gym
```

- ASAP
```
https://github.com/LeCAR-Lab/ASAP
```

- RLinf https://github.com/RLinf/RLinf , 4.3k star, 高性能架构，似乎是计算机系出的一个开源代码。

- musclemimic
```
[2026]Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale
https://github.com/amathislab/musclemimic
```

- MyoSuite
```
1. MyoSuite is a collection of environments/tasks to be solved by musculoskeletal models simulated with the MuJoCo physics engine and wrapped in the OpenAI gym API.
2. paper MyoSuite2022, MyoSuite -- A contact-rich simulation suite for musculoskeletal motor control,
3. updated 2 months ago
4. github address, https://github.com/myohub/myosuite, 1.2k star
```

- musclemimic_models
```
1. Musculoskeletal models released from musclemimic research: fullbody musculoskeletal model
https://github.com/amathislab/musclemimic_models, 53 star
```


- are there any research paper or open source code framework related to general purpose robot behavior foundation model?
```
pi0 灵巧手通用策略模型 https://www.pi.website/blog/pi0

```





- Strand based hair rendering
```
https://github.com/AEspinosaDev/Three-Hair
https://github.com/Jhonve/HairStrandsRendering
https://github.com/AEspinosaDev/WebGL-RealTimeFur-SEDDI
https://github.com/Scthe/frostbitten-hair-webgpu
https://github.com/digital-salon/Digital-Salon
```

