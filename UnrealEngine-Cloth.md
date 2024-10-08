# 练习UE5中如何制作布料

- 布料制作工具 VRoid Studio 1.29
- UE5.4 Chaos Cloth插件 https://www.youtube.com/watch?v=QnHOjRDQtfo
- KawaiiPhysics插件 https://pafuhana1213.hatenablog.com/entry/2019/07/26/171046
- VRM格式导入插件 https://www.3dcat.live/share/post-id-226/

## 下载Cloth.7z 
Learn https://dev.epicgames.com/community/unreal-engine/learning?

从UE官网->学习->学习资料库 搜索"Cloth" 找到“Panel Cloth Example Files (5.4)”， 下载 UE-5-4-ChaosCloth.7z （ https://d1iv7db44yhgxn.cloudfront.net/post-static-files/UE-5-4-ChaosCloth.7z ）

重命名为Cloth.7z， 解压，以这个项目作为起始项目。

以这个项目作为基础。

## 下载VRM4U插件

从[VRM4U](https://github.com/ruyo/VRM4U/releases)下载VRM4U插件，放到Cloth文件夹的Plugins目录中，修改Cloth.uproject，添加VRM4U插件
```
{
  "Name": "VRM4U",
  "Enabled": true,
  "SupportedTargetPlatforms": [
    "Win64"
  ]
}
```

- 编译源码
VRM4U/Source/ReleaseScript中有脚本可以将源代码编译为当前UE版本。需要适当修改脚本。 把执行ps1文件的部分去掉。

## RagDoll 布偶
- Introduction
```
在物理模拟中，Ragdoll约束（Ragdoll physics）是一种模拟人物或物体在受到外力作用时，如何自然地倒下或移动的技术。这种技术通常用于游戏和动画制作中，以创建更加逼真的物理反应。
"Ragdoll"这个名字来源于这种物理模拟的直观表现，即模拟对象（通常是人物角色）在受到冲击或失去支撑时，会像布偶（rag doll）一样放松，其四肢和身体会自然下垂并按照物理定律进行运动。以下是Ragdoll约束的主要特点：
关节约束：在Ragdoll模拟中，各个关节（如肩膀、肘部、膝盖等）被设置为可以有限度地旋转的约束。这些约束模拟了真实人体的关节运动限制。
软性碰撞：Ragdoll的各个部分在碰撞检测中通常被设定为有一定弹性，使得碰撞看起来更加自然。
重力影响：Ragdoll模拟中的物体受到重力的影响，当角色倒下时，其动作会因重力作用而显得更加真实。
动态响应：当Ragdoll受到外力作用时，如被击打或推动，其反应会根据力的方向和大小动态变化。
物理计算：Ragdoll的动态行为是通过物理引擎实时计算得出的，这包括碰撞检测、力的应用、质量、摩擦和空气阻力等因素。
Ragdoll约束在游戏开发中特别有用，因为它可以创建出非常逼真的场景，比如角色从高处跌落、被击中后倒下等，而不需要逐帧手动调整动画，大大提高了游戏的真实感和开发效率。然而，为了确保模拟的真实性和游戏体验的流畅性，开发者需要仔细调整Ragdoll参数，以防止不自然的动作或物理表现。
```

