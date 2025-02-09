# 创建交互体验

- [1.人工智能](#1人工智能)
- [2.Gameplay相机系统](#2Gameplay相机系统)
- [3.Gameplay定位系统](#3Gameplay定位系统)
- [4.输入](#4输入)
- [5.联网和多人游戏](#5联网和多人游戏)


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
- PlayerInput
- PlayerInputComponent
- ActionMapping 将离散按钮或按键映射到一个"友好的名称"，该名称稍后将与事件驱动型行为绑定
- AxisMapping for 连续行为，会做轮询
- 项目设置 -> 输入 -> Action Mapping
- 项目设置 -> 输入 -> Axis Mapping
- InputComponent这个跟C++中Component的概念不同，指的是Actor, PlayerController, Level Blueprint, Pawn都可以处理Input, 所以有优先级顺序。
![InputComponent](https://d1iv7db44yhgxn.cloudfront.net/documentation/images/f5592819-a96b-4384-b9fe-b38fe5494942/inputflow.png)
```
void AFirstPersonBaseCodeCharacter::SetupPlayerInputComponent(class UInputComponent* InputComponent)
{
   // set up gameplay key bindings
   check(InputComponent);
   ...
   InputComponent->BindAxis("MoveForward", this, &AFirstPersonBaseCodeCharacter::MoveForward);
   ...
}
void AFirstPersonBaseCodeCharacter::MoveForward(float Value)
{
    if ( (Controller != NULL) && (Value != 0.0f) )
    {
        // find out which way is forward
        FRotator Rotation = Controller->GetControlRotation();
        // Limit pitch when walking or falling
        if ( CharacterMovement->IsMovingOnGround() || CharacterMovement->IsFalling() )
        {
            Rotation.Pitch = 0.0f;
        }
        // add movement in that direction
        const FVector Direction = FRotationMatrix(Rotation).GetScaledAxis(EAxis::X);
        AddMovementInput(Direction, Value);
    }
}
```
- 触摸界面
  - DefaultVirtualJoysticks
  - LeftVirtualJoystickOnly

- 增强输入插件 Enhanced Input
  - **输入操作 Input Action asset**
  - **输入映射上下文 Input Mapping Context**
APawn::SetupPlayerInputComponent中BindAction
```
// 确保我们正在使用 UEnhancedInputComponent；如果未使用，表明项目未正确配置。
if (UEnhancedInputComponent* PlayerEnhancedInputComponent = Cast<UEnhancedInputComponent>(PlayerInputComponent))
{
  // 有多种方式可以将UInputAction*绑定到处理程序函数和可能有关的多种类型的ETriggerEvent。

  // 当MyInputAction启动时，这会在更新函数上调用处理程序函数，例如在按操作按钮时。
  if (MyInputAction)
  {
      PlayerEnhancedInputComponent->BindAction(MyInputAction, ETriggerEvent::Started, this, &AMyPawn::MyInputHandlerFunction);
  }

  // 当满足输入条件时，例如在按住某个动作键时，这会在每个更新函数上按名称调用处理程序函数(UFUNCTION)。
  if (MyOtherInputAction)
  {
      PlayerEnhancedInputComponent->BindAction(MyOtherInputAction, ETriggerEvent::Triggered, this, TEXT("MyOtherInputHandlerFunction"));
  }
}
```

- 可以设置多个输入映射上下文。
比如开车一个，游泳一个，行走一个。
填充输入映射上下文之后，就可以将其添加到与Pawn的玩家控制器关联的本地玩家。通过重载PawnClientRestart函数和添加代码块可以完成此目标，如下所示：
```
// 确保我们具有有效的玩家控制器。
if (APlayerController* PC = Cast<APlayerController>(GetController()))
{
  // 从与我们的玩家控制器相关的本地玩家获取Enhanced Input本地玩家子系统。
  if (UEnhancedInputLocalPlayerSubsystem* Subsystem = ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(PC->GetLocalPlayer()))
  {
      // PawnClientRestart在Actor的生命周期中可以运行多次，因此首先要清除任何残留的映射。
      Subsystem->ClearAllMappings();

      // 添加每个映射上下文及其优先级值。高值的优先级高于低值。
      Subsystem->AddMappingContext(MyInputMappingContext, MyInt32Priority);
  }
}
```

## 5.联网和多人游戏
### 5.1.网络多人游戏基础

#### 网络概述
**与本地多人游戏不同，这带来了额外的挑战。不同的客户端可能有不同的网络连接速度，而且信息必须通过网络来传达，但是网络有可能不稳定，输入可能会丢失。因此，在任何给定时间，一台客户端计算机上的游戏状态很可能不同于其他每一台客户端计算机。服务器作为游戏主机保存一个真正的 权威 游戏状态。换句话说，服务器是多人游戏实际运行的地方。客户端各自控制它们在服务器上拥有的远程 Pawn 。客户端从其本地Pawn向其服务器Pawn发送 远程程序调用 以在游戏中执行操作。接着，服务器向每个客户端 复制 关于游戏状态的信息，例如 Actor 所在的位置，这些Actor应该具备怎样的行为，以及不同的变量应该有哪些值。然后每个客户端使用这些信息，近似模拟服务器上实际正在发生的情况。**

关键字： 游戏状态信息： 比如Actor的位置，Acor应该具备怎样的行为，不同的变量的值。

服务器向每个客户端复制关于游戏状态的信息。 一般不将视觉效果流送到客户端显示器进行显示。

#### **客户端-服务器Gameplay示例**

```
玩家1在本地计算机上按输入以发射武器。
玩家1的本地Pawn将发射武器的命令中传递到服务器上其连接的Pawn。
玩家1在服务器上的武器生成发射物。
服务器通知每个连接的客户端各自创建玩家1的发射物副本。
玩家1在服务器上的武器通知每个客户端播放与发射武器关联的声音和视觉效果。

玩家1在服务器上的发射物从武器中射出。
服务器通知每个客户端复制玩家1的发射物正在进行的移动，以便每个客户端上的玩家1发射物也跟着移动。

玩家1在服务器上的发射物撞击玩家2的Pawn。

该撞击针触发一个函数来销毁玩家1在服务器上的发射物。
服务器自动通知每个客户销毁玩家1发射物的副本。
该撞击将触发一个函数来通知所有客户端播放该撞击随附的声音和视觉效果。
玩家2在服务器上的Pawn承受因发射物撞击造成的伤害。
玩家2在服务器上的Pawn通知玩家2的客户端播放画面效果响应该伤害。
```
在本地多人游戏中，这些交互全部在同一台计算机上的同一个世界中发生，因此相较于联网多人游戏更易于理解和编程。例如，当游戏生成一个对象时，所有玩家肯定都能看到该对象，因为所有玩家全部存在于同一个游戏世界中。

而在联网多人游戏中，这些交互在多个不同的世界中发生：
服务器上的权威世界。
玩家1的客户端世界。
玩家2的客户端世界。
连接到此服务器游戏实例的其他所有客户端的额外世界。

每个世界都有自己的玩家控制器、Pawn、武器和发射物。服务器是游戏实际运行的地方，但每个客户端的世界必须准确复制服务器上发生的事件，因此需要向每个客户端选择性地发送信息，在视觉上准确地展示服务器上的世界。

此过程在基础Gameplay交互（碰撞、移动、损伤）、美化效果（视觉效果和声音）以及玩家信息（HUD更新）之间进行了划分。这三者各自与网络中的特定计算机或一组计算机关联。**此信息的复制过程并非完全自动化，你必须在Gameplay编程中指定要将哪些信息复制到哪些计算机。主要难点在于选择应该将哪些信息复制到哪些连接，才能为所有玩家提供一致的体验，同时还要最大限度减少信息复制量，避免网络带宽频繁饱和。**

#### 基础网络概念
##### 网络模式
我们将用网络模式 ENetMode::NM_DedicatedServer实现专用服务器

##### 复制
主要使用Actor和Actor派生的类通过UE中的网络连接复制其状态。AActor 是可以在关卡中放置或生成的对象的基类，也是UE的 UObject 继承层级中第一个支持用于网络的类。UObject 派生的类也可以复制，但必须作为复制的子对象附加到Actor才能恰当复制。

**Actor复制** 和 **复制系统**


###### Actor复制


#### 复制系统





Gameplay框架 https://dev.epicgames.com/documentation/zh-cn/unreal-engine/gameplay-framework-in-unreal-engine?application_version=5.4
Gameplay技能系统 https://dev.epicgames.com/documentation/zh-cn/unreal-engine/gameplay-ability-system-for-unreal-engine?application_version=5.4
物理 https://dev.epicgames.com/documentation/zh-cn/unreal-engine/physics-in-unreal-engine?application_version=5.4
碰撞 https://dev.epicgames.com/documentation/zh-cn/unreal-engine/collision-in-unreal-engine?application_version=5.4
