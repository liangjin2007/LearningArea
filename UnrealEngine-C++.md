# [用C++编程](https://dev.epicgames.com/documentation/en-us/unreal-engine/programming-with-cplusplus-in-unreal-engine) 
- [Actor](#1AActor)
- 组件Components
- 反射系统
- 游戏性架构
- 虚幻引擎中的容器
- 委托
- 布料模拟
- 代码规范
- 其他


## 1.AActor
```
在Unreal Engine中，AActor 类是所有游戏世界中的实体的基类。
它提供了基本的属性和行为，如位置、旋转和缩放。
以下是 AActor 的一些常用派生类，这些派生类在游戏开发中经常被使用：
APawn：
用于表示可以由玩家或AI控制的角色。
	进一步派生类：
		ACharacter：具有角色特定功能的Pawn，如第三人称或第一人称角色。
		APlayerController：玩家控制的Pawn，负责处理玩家的输入。
AStaticMeshActor：
用于表示静态网格物体，如环境中的建筑、道具等。
ADecalActor：
用于表示贴花，如在地面上绘制的痕迹或标志。
AEmitterActor：
用于表示粒子发射器，通常用于创建视觉效果，如烟雾、火等。
ALightActor：
用于表示场景中的光源。
ACameraActor：
用于定义摄像机位置和属性，通常与UCameraComponent一起使用。
AInfo：
用于存储游戏信息，不直接渲染，但可以用于存储数据或作为游戏逻辑的一部分。
AHUD：
虽然通常不直接作为世界中的实体，但它是AActor的派生类，用于渲染和控制游戏UI。
APlayerState：
用于存储玩家的状态信息，如分数、生命值等。
AGameStateBase：
用于存储游戏的全局状态信息。
AController：
用于表示控制Pawn的逻辑实体，可以是玩家或AI。
ABP_BaseCharacter：
这通常是由蓝图创建的Character基类，包含了开发者设定的基础角色功能。
ABP_BasePawn：
类似于ABP_BaseCharacter，但是用于基于蓝图的Pawn基类。

这些派生类提供了不同的功能，以适应不同的游戏设计需求。
开发者可以根据需要进一步派生这些类，或者直接使用它们来创建游戏世界中的各种实体。
此外，许多游戏还会创建自定义的Actor类来满足特定的游戏需求。
```



## 2.组件
```
组件是一种特殊类型的对象，Actor 可以将组件作为子对象附加到自身。
组件适用于共享相同的行为，例如显示视觉表现、播放声音。
它们还可以表示项目特有的概念，例如载具解译输入和改变其速度与方向的方式。
举例而言，某个项目拥有用户可控制车辆、飞机和船只。
可以通过更改载具Actor所使用的组件来实现载具控制和移动的差异。
```
- Actor组件 ActorComponent
```
UActorComponent 是所有组件的基类。
由于组件是渲染网格体和图像、实现碰撞和播放音频的唯一方法，
因此玩家游戏期间在场景中看到或进行交互的一切其实都是某一类组件的成果。
Unreal Engine（虚幻引擎）中的UActorComponent类是所有组件的基类，用于附加到AActor（代表游戏世界中的实体）的基类。
组件负责实现特定的功能，如渲染、物理交互、声音播放等。
以下是UActorComponent的一些常用派生类，这些派生类在游戏开发中经常被使用：
UPrimitiveComponent：
	UStaticMeshComponent：用于渲染静态网格。
	UDecalComponent：用于渲染贴花，比如在地面上显示血迹。
	UCollisionShapeComponent：用于定义碰撞体，通常与物理交互相关。
	UBoxComponent：用于创建盒形碰撞体。
	USphereComponent：用于创建球形碰撞体。
	UCapsuleComponent：用于创建胶囊形碰撞体。
USkeletalMeshComponent：
用于渲染骨骼网格，通常与动画相关。
UStaticMeshComponent：
用于渲染不动的物体，如环境中的建筑和道具。
ULightComponent：
用于表示场景中的光源。
UCameraComponent：
用于定义玩家或AI的视角。
UAudioComponent：
用于播放音效或音乐。
USplineComponent：
用于创建和渲染样条线，常用于路径规划和动画。
UTextRenderComponent：
用于渲染文本，比如游戏中的分数或提示信息。
UWidgetComponent：
用于将UI元素渲染到游戏世界中。
UMaterialBillboardComponent：
用于创建基于材质的公告牌，通常用于快速渲染粒子效果。
UParticleSystemComponent：
用于创建和渲染粒子系统。
UForceFeedbackComponent：
用于实现游戏手柄的震动效果。
这些组件可以附加到任何AActor派生的类上，为游戏对象提供各种功能。
在虚幻引擎中，通过组合这些不同的组件，可以创建出具有复杂行为和功能的游戏实体。
```

```
注册组件: 自动注册与手工注册，RegisterComponent, 游戏运行期间注册可能会影响性能，谨慎使用。
	注册事件：
		UActorComponent::OnRegister
		UActorComponent::CreateRenderState
		UActorComponent::OnCreatePhysicsState
取消注册组件: UnregisterComponent
	取消注册事件:
		UActorComponent::OnUnregister
		UActorComponent::DestroyRenderState
		UActorComponent::OnDestroyPhysicsState
更新
	类似于Actor的逐帧更新，组件也能逐帧更新。UActorComponent::TickComponent()。
	例如，USkeletalMeshComponent 使用其 TickComponent 函数来更新动画和骨架控制器。
	而UParticleSystemComponent 更新其发射器和处理粒子事件。
	默认情况下，Actor组件不更新。为了让Actor组件逐帧更新，必须在构造函数中将 PrimaryComponentTick.bCanEverTick 设置为 true 来启用tick。
	之后，在构造函数中或其他位置处，必须调用 PrimaryComponentTick.SetTickFunctionEnable(true) 以开启更新。
	之后可调用 PrimaryComponentTick.SetTickFunctionEnable(false) 停用tick。
	如果您知道组件永远不需要更新，或者打算手动调用自己的更新函数（也许从拥有的Actor类），将 PrimaryComponentTick.bCanEverTick 保留为默认值 false 即可，这样可以稍微改善性能。
渲染状态
	为进行渲染，Actor组件必须创建渲染状态。
	此渲染状态还会告诉引擎，需要更新渲染数据的组件已发生变更。
	当发生此类变更时，渲染状态会被标记为"dirty"。
	如果编译您自己的组件，可以使用 MarkRenderStateDirty 函数将渲染数据标记为dirty。
	在一帧结束时，所有dirty组件的渲染数据都会在引擎中更新。
	场景组件（包括Primitive组件）默认会创建渲染状态，而Actor组件则不会。
	如何写？
物理状态
	要与引擎的物理模拟系统交互，Actor组件需要物理状态。
	物理状态会在发生变化时立即更新，防止出现"帧落后"瑕疵等问题，也不需要"dirty"标记。
	默认情况下，Actor组件和场景组件没有物理状态，但基元组件有。
	覆盖 ShouldCreatePhysicsState 函数以确定组件类实例是否需要物理状态。
	如果类使用物理，则不建议只返回 true。
	请参阅函数的 UPrimitiveComponent 版本，了解不应创建物理状态的情况（例如在组件破坏期间）。
	在正常返回 true 的情况下，还可以返回 Super::ShouldCreatePhysicsState。
视觉化组件
	
```  
- 场景组件 SceneComponent
```
场景组件是指存在于场景中特定物理位置处的Actor组件。
该位置由 变换（类 FTransform）定义，其中包含组件的位置、旋转和缩放。
场景组件能够通过将彼此连接起来形成树，Actor可以将单个场景组件指定为"根"，意味着这个Actor的场景位置、旋转和缩放都根据此组件来绘制。
总结起来就是有位置和层次。
```  
```
附加:
只有场景组件（USceneComponent 及其子类）可以彼此附加，因为需要变换来描述子项和父项组件之间的空间关系。
虽然场景组件可以拥有任意数量的子项，但只能拥有一个父项，或可直接放置在场景中。
场景组件系统不支持附加循环。
两种主要方法分别是 SetupAttachment 和 AttachToComponent。
前者在构造函数中、以及处理尚未注册的组件时十分实用；后者会立即将场景组件附加到另一个组件，在游戏进行中十分实用。
该附加系统还允许将Actor彼此之间进行附加，方法是将一个Actor的根组件附加到属于另一个Actor的组件。
```

- 基元组件 PrimitiveComponent
```
基元组件（类 UPrimitiveComponent）是包含或生成某类几何的场景组件，通常用于渲染或碰撞。
各种类型的几何体，目前最常用的是 盒体组件、胶囊体组件、静态网格体组件 和 骨架网格体组件。
盒体组件和胶囊体组件生成不可见的几何体进行碰撞检测，而静态网格体组件和骨架网格体组件包含将被渲染的预制几何体，需要时也可以用于碰撞检测。
```
```
场景代理:
基元组件的 场景代理（类 FPrimitiveSceneProxy）封装场景数据，引擎使用这些数据来与游戏线程并行渲染组件。
每种类型的基元都有自身的场景代理子类，用来保存所需的特定渲染数据。

```  


# 反射系统
章节
- 对象
```
UCLASS宏
属性和函数
UObject创建
	UObjects 不支持构造器参数。所有的C++ UObject都会在引擎启动的时候初始化，然后引擎会调用其默认构造器。如果没有默认的构造器，那么 UObject 将不会编译。
	
	UObject 构造器应该轻量化，仅用于设置默认的数值和子对象，构造时不应该调用其它功能和函数。对于 Actor和Actor组件，初始化功能应该输入 BeginPlay() 方法。
	
	UObject 应该仅在运行时使用 NewObject 构建，或者将 CreateDefaultSubobject 用于构造器。

	UObjects 永远都不应使用 new 运算符。所有的 UObjects 都由虚幻引擎管理内存和垃圾回收。如果通过 new 或者 delete 手动管理内存，可能会导致内存出错。

UObjects 提供的功能
	此系统的使用不为强制要求，甚至有时不适合使用，但却存在以下益处：

		垃圾回收
		引用更新
		反射
		序列化
		默认属性变化自动更新
		自动属性初始化
		自动编辑器整合
		运行时类型信息可用
		网络复制

	大部分这些益处适用于 UStruct，它有着和 UObject 一样的反射和序列化能力。UStruct 被当作数值类型处理并且不会垃圾回收。

虚幻头文件工具
	为利用 UObject 派生类型所提供的功能，需要在头文件上为这些类型执行一个预处理步骤，以核对需要的信息。 该预处理步骤由 UnrealHeaderTool（简称 UHT）执行。UObject 派生的类型需要遵守特定的结构。

头文件格式
	UObject 在源（.cpp）文件中的实现与其他 C++ 类相似，其在头（.h）文件中的定义必须遵守特定的基础结构，以便在虚幻引擎 4 中正常使用。使用编辑器的"New C++ Class"命令是设置格式正确头文件的最简单方法。UObject 派生类的基础头文件可能看起来与此相似，假定 UObject 派生物被称为 UMyObject，其创建时所在的项目被称为 MyProject：

	#pragma once
 
	#include 'Object.h'
	#include 'MyObject.generated.h'
 
	UCLASS()
	class MYPROJECT_API UMyObject : public UObject
	{
		GENERATED_BODY()
 
	};

	如 MyProject 希望将 UMyObject 类公开到其他模块，则需要指定 MYPROJECT_API。这对游戏项目将包括的模块或插件十分实用。这些模块和插件将故意使类公开，在多个项目间提供可携的自含式功能。

更新对象
	Ticking 代表虚幻引擎中对象的更新方式。所有Actors均可在每帧被 tick，便于您执行必要的更新计算或操作。

	Actor 和 Actor组件在注册时会自动调用它们的 Tick 函数，然而，UObjects 不具有嵌入的更新能力。在必须的时候，可以使用 inherits 类说明符从 FTickableGameObject 继承即可添加此能力。 这样即可实现 Tick() 函数，引擎每帧都将调用此函数。

销毁对象
	对象不被引用后，垃圾回收系统将自动进行对象销毁。这意味着没有任何 UPROPERTY 指针、引擎容器、TStrongObjectPtr 或类实例会拥有任何对它的强引用。

	注意，无论对象是否被垃圾回收，弱指针对其都没有影响。
	
	垃圾回收器运行时，寻找到的未引用对象将被从内存中移除。此外，函数MarkPendingKill() 可在对象上直接调用。此函数将把指向对象的所有指针设为 NULL，并从全局搜索中移除对象。对象将在下一次垃圾回收过程中被完全删除。
	
	智能指针不适用于 UObject。
	
	Object->MarkPendingKill() 被 Obj->MarkAsGarbage() 所替代。这个新的函数现在仅用于追踪旧对象。如果 gc.PendingKillEnabled=true ，那么所有标记为 PendingKill 的对象会被垃圾回收器自动清空并销毁。
	
	强引用会将 UObject 保留。如果你不想让这些引用保留 UObject，那么这些引用应该转换来使用弱指针，或者变为一个普通指针由程序员手动清除（如果对性能有较高要求的话）。
	
	你可以用弱指针替换强指针，并且在游戏运作时作为垃圾回收取消引用，因为垃圾回收仅在帧之间运行。
	
	IsValid() 用于检查它是 null 还是垃圾，但是大部分情况下 IsValid 可以被更正规的编程规则替换，比如在调用 OnDestroy 事件时将指针清除至 Actor。
	
	如果禁用了 PendingKill()， MarkGarbage() 将会提醒对象的所有者该对象将要被销毁，但是对象本身直到所有对它的引用都解除之后才会被垃圾回收。
	
	对于 Actor，即使 Actor 被调用了 Destroy()，并且被从关卡中移除，它还是会等到所有对它的引用都解除之后才会被垃圾回收。
	
	对于证书持有者的主要区别在于，对花费较大的对象进行垃圾回收的函数 MarkPendingKill() 不再起效。
	
	已有的用于 nullptr 的检查应该被 IsValid() 调用所替代，除非你进行手动清除，因为指针不再会被垃圾回收器通过 MarkPendingKill() 自动清除。

```

- 属性
```
属性声明：
	UPROPERTY([specifier, specifier, ...], [meta(key=value, key=value, ...)])
	Type VariableName;
核心整数类型：
	整数
		int8, uint8, int16, uint16, int32, uint32, int64, uint64
	作为掩码 : https://dev.epicgames.com/documentation/zh-cn/unreal-engine/unreal-engine-uproperties
		1.添加元标记即可 UPROPERTY(EditAnywhere, Meta = (Bitmask)) int32 BasicBits;
		添加此元标记将使整数作为下拉列表形式可供编辑，它们使用笼统命名标记（"Flag 1"、"Flag 2"、"Flag 3"等等），可以 单独打开或关闭。
		2.UFUNCTION(BlueprintCallable) void MyFunction(UPARAM(meta=(Bitmask)) int32 BasicBitsParam);
		3.为了让Flag 1这种名称转为自定义名称： 为了自定义位标记名称，首先必须使用"bitflags"元标记来创建UENUM：
			UENUM(Meta = (Bitflags))
			enum class EColorBits
			{
				ECB_Red,
				ECB_Green,
				ECB_Blue
			};
			比特掩码枚举类型的范围是0到31，包括0和31。其对应于32位整型变量的位数（从第0位开始）。在上面的例子中，第0位是 ECB_Red，第1位是 ECB_Green，第2位是 ECB_Blue。
			...

浮点数：
	float, double

布尔类型：
	bool bIsThirsty;
	int32 bIsHungry: 1;

字符串
	三种核心类型： FString, FName, FText
	FString是典型的"动态字符数组"字符串类型。
	FName是对全局字符串表中不可变且不区分大小写的字符串的引用。相较于FString，它的大小更小，更能高效的传递，但更难以操控。
	FText是指定用于处理本地化的更可靠的字符串表示。
	大多数情况下，虚幻依靠TCHAR类型来表示字符， TEXT()宏可用于表示TCHAR文字。 MyDogPtr->DogName = FName(TEXT("Samson Aloysius"));
	
属性说明符（specifier）

Metadata 说明符(metadata specifier)
	
```



- 结构体
```
结构体（Struct） 是一种数据结构，帮助你组织和操作相关属性。在虚幻引擎中，结构体会被引擎的反射系统识别为 UStruct，但不属于 UObject生态圈,且不能在UClasses的内部使用。

在相同的数据布局下， UStruct 比 UObject 能更快创建。

UStruct支持UProperty, 但它不由垃圾回收系统管理，不能提供UFunction。

实现USTRUCT

结构体说明符
	Atomic	表示该结构体应始终被序列化为一个单元。将不会为该类创建自动生成的代码。标头仅用于解析元数据。
	BlueprintType	将此结构体作为一种类型公开，可用于蓝图中的变量。
	NoExport	将不会为该类创建自动生成的代码。标头仅用于解析元数据。

最佳做法与技巧
	下面是一些使用 UStruct 时需要记住的有用提示：
	
	UStruct 可以使用虚幻引擎的智能指针和垃圾回收系统来防止垃圾回收删除 UObjects。
	
	结构体最好用于简单数据类型。对于你的项目中更复杂的交互，也许可以使用 UObject 或 AActor 子类来代替。
	
	UStructs 不可以 用于复制。但是 UProperty 变量 可以 用于复制。
	
	虚幻引擎可以自动为结构体创建Make和Break函数。
	
	Make函数出现在任何带有 BlueprintType 标签的 Ustruct 中。
	如果在UStruct中至少有一个 BlueprintReadOnly 或 BlueprintReadWrite 属性，Break函数就会出现。
	Break函数创建的纯节点为每个标记为 BlueprintReadOnly 或 BlueprintReadWrite 的资产提供一个输出引脚。


```
- TSubclassOf
```
	UClass* ClassA = UDamageType::StaticClass();
 
	TSubclassOf<UDamageType> ClassB;
 
	ClassB = ClassA; // Performs a runtime check
 
	TSubclassOf<UDamageType_Lava> ClassC;
 
	ClassB = ClassC; // Performs a compile time check

```


- 接口
```
接口声明
接口说明符
在C++中实现接口
声明接口函数
	仅C++的接口函数
	蓝图可调用接口函数
		BlueprintCallable
		BlueprintImplementableEvent
		BlueprintNativeEvent
确定类是否实现了接口
	bool bIsImplemented = OriginalObject->GetClass()->ImplementsInterface(UReactToTriggerInterface::StaticClass()); // 如果OriginalObject实现了UReactToTriggerInterface，则bisimplemated将为true。
 
	bIsImplemented = OriginalObject->Implements<UReactToTriggerInterface>(); // 如果OriginalObject实现了UReactToTrigger，bIsImplemented将为true。
 
	IReactToTriggerInterface* ReactingObject = Cast<IReactToTriggerInterface>(OriginalObject); // 如果OriginalObject实现了UReactToTriggerInterface，则ReactingObject将为非空。
转换到其他虚幻类型
	IReactToTriggerInterface* ReactingObject = Cast<IReactToTriggerInterface>(OriginalObject); // 如果接口被实现，则ReactingObject将为非空。
 
	ISomeOtherInterface* DifferentInterface = Cast<ISomeOtherInterface>(ReactingObject); // 如果ReactingObject为非空而且还实现了ISomeOtherInterface，则DifferentInterface将为非空。
 
	AActor* Actor = Cast<AActor>(ReactingObject); // 如果ReactingObject为非空且OriginalObject为AActor或AActor派生的类，则Actor将为非空。
蓝图可实现类
	如果你想要蓝图能够实现此接口，则必须使用"Blueprintable"元数据说明符。蓝图类要覆盖的每个接口函数都必须是"BlueprintNativeEvent"或"BlueprintImplementableEvent"。标记为"BlueprintCallable"的函数仍然可以被调用，但不能被覆盖。你将无法从蓝图访问所有其他函数。
```

- 元数据说明符 metadata specifier
```

```


- UFunction
```
UFunction 是虚幻引擎（UE）反射系统可识别的C++函数。UObject 或蓝图函数库可将成员函数声明为UFunction，方法是将 UFUNCTION 宏放在头文件中函数声明上方的行中。宏将支持 函数说明符 更改虚幻引擎解译和使用函数的方式。

UFUNCTION([specifier1=setting1, specifier2, ...], [meta(key1="value1", key2, ...)])
[static] ReturnType FunctionName([Parameter1, Parameter2, ..., ParameterN1=DefaultValueN1, ParameterN2=DefaultValueN2]) [const];

可利用函数说明符将UFunction对蓝图可视化脚本图表公开，以便开发者从蓝图资源调用或扩展UFunction，而无需更改C++代码。

在类的默认属性中，UFunction可绑定到委托，从而能够执行一些操作（例如将操作与用户输入相关联）。

它们还可以充当网络回调，这意味着当某个变量受网络更新影响时，用户可以将其用于接收通知并运行自定义代码。

用户甚至可创建自己的控制台命令（通常也称 debug、configuration 或 cheat code 命令），并能在开发版本中从游戏控制台调用这些命令，或将拥有自定义功能的按钮添加到关卡编辑器中的游戏对象。

```



- 智能指针库
```
共享指针（TSharedPtr）	共享指针拥有其引用的对象，无限防止该对象被删除，并在无共享指针或共享引用（见下文）引用其时，最终处理其的删除。共享指针可为空白，意味其不引用任何对象。任何非空共享指针都可对其引用的对象生成共享引用。
共享引用（TSharedRef）	共享引用的行为与共享指针类似，即其拥有自身引用的对象。对于空对象而言，其存在不同；共享引用须固定引用非空对象。共享指针无此类限制，因此共享引用可固定转换为共享指针，且该共享指针固定引用有效对象。要确认引用的对象是非空，或者要表明共享对象所有权时，请使用共享引用。
弱指针（TWeakPtrTSharedPtr）	弱指针类与共享指针类似，但不拥有其引用的对象，因此不影响其生命周期。此属性中断引用循环，因此十分有用，但也意味弱指针可在无预警的情况下随时变为空。因此，弱指针可生成指向其引用对象的共享指针，确保程序员能对该对象进行安全临时访问。
唯一指针（TUniquePtr）	唯一指针仅会显式拥有其引用的对象。仅有一个唯一指针指向给定资源，因此唯一指针可转移所有权，但无法共享。复制唯一指针的任何尝试都将导致编译错误。唯一指针超出范围时，其将自动删除其所引用的对象。


类	 
	TSharedFromThis	在添加 AsShared 或 SharedThis 函数的 TSharedFromThis 中衍生类。利用此类函数可获取对象的 TSharedRef。
函数	 
	MakeShared 和 MakeShareable	在常规C++指针中创建共享指针。MakeShared 会在单个内存块中分配新的对象实例和引用控制器，但要求对象提交公共构造函数。MakeShareable 的效率较低，但即使对象的构造函数为私有，其仍可运行。利用此操作可拥有非自己创建的对象，并在删除对象时支持自定义行为。
	StaticCastSharedRef 和 StaticCastSharedPtr	静态投射效用函数，通常用于向下投射到衍生类型。
	ConstCastSharedRef 和 ConstCastSharedPtr	将 const 智能引用或智能指针分别转换为 mutable 智能引用或智能指针。
```



# 游戏性架构

# 容器
- TMap TMultiMap
```
创建和填充Map
	TMap<int32, FString> a;
	a.Add(key, value); // 类似于std::map::insert
	a.Add(key); // 值会被默认构造
	a.Emplace(key, value);
	
	TMap<int32, FString> b;
	...
	b.Append(a);
	// Note: a is empty now
	
	如果用 UPROPERTY 宏和一个可编辑的关键词（EditAnywhere、EditDefaultsOnly 或 EditInstanceOnly）标记 TMap，即可在编辑器中添加和编辑元素。
	UPROPERTY(Category = MapsAndSets, EditAnywhere)
	TMap<int32, FString> FruitMap;


迭代Iterator
	for (auto It = FruitMap.CreateConstIterator(); It; ++It)
	{
		It.Key()
		It.Value()
	};

查询
	a.Contains(k);
	a.Num()
	a[7] // 在使用[]operator之前，先要判断是否包含这个元素
	FString* Ptr7 = FruitMap.Find(7);
	FString& Ref7 = FruitMap.FindOrAdd(7);
	FindRef: 不要被名称迷惑，FindRef 会返回与给定键关联的值副本；若映射中未找到给定键，则返回默认构建值。FindRef 不会创建新元素，因此既可用于常量映射，也可用于非常量映射。
	FindKey 函数执行逆向查找，这意味着提供的值与键匹配，并返回指向与所提供值配对的第一个键的指针。搜索映射中不存在的值将返回空键。
		按值查找比按键查找慢（线性时间）。这是因为映射是根据键而不是值进行哈希。此外，如果映射有多个具有相同值的键，FindKey 可返回其中任一键。
	a.GenerateKeyArray(outKeyArray);
	a.GenerateValueArray(outValueArray);

移除
	a.Remove(key);
	FString Removed7 = FruitMap.FindAndRemoveChecked(7); // FindAndRemoveChecked 函数可用于从映射移除元素并返回其值。名称的"已检查"部分表示若键不存在，映射将调用 check（UE4中等同于 assert）。
	FString Removed8 = FruitMap.FindAndRemoveChecked(8); // Assert!

	FString Removed;
	bool bFound2 = FruitMap.RemoveAndCopyValue(2, Removed); //RemoveAndCopyValue 函数的作用与 Remove 相似，不同点是会将已移除元素的值复制到引用参数。如果映射中不存在指定的键，则输出参数将保持不变，函数将返回 false。

	a.Empty();
	a.Reset();

排序
	FruitMap.KeySort([](int32 A, int32 B) {
		return A > B; // sort keys in reverse
	});

	FruitMap.ValueSort([](const FString& A, const FString& B) {
		return A.Len() < B.Len(); // sort strings by length
	});

运算符
	可通过复制构造函数，赋值运算符进行复制。深层拷贝，拥有各自的元素。

	支持移动语义： a.MoveTemp(b);  // b为空， a变成b。

Slack
	FruitMap.Reserve(10);
	for (int32 i = 0; i < 10; ++i)
	{
		FruitMap.Add(i, FString::Printf(TEXT("Fruit%d"), i));
	}


	for (int32 i = 0; i < 10; i += 2)
	{
		FruitMap.Remove(i);
	}
	// FruitMap == [
	// 	{ Key: 9, Value: "Fruit9" },
	// 	<invalid>,
	// 	{ Key: 7, Value: "Fruit7" },
	// 	<invalid>,
	// 	{ Key: 5, Value: "Fruit5" },
	// 	<invalid>,
	// 	{ Key: 3, Value: "Fruit3" },
	// 	<invalid>,
	// 	{ Key: 1, Value: "Fruit1" },
	// 	<invalid>
	// ]
	FruitMap.Shrink(); 
	// FruitMap == [
	// 	{ Key: 9, Value: "Fruit9" },
	// 	<invalid>,
	// 	{ Key: 7, Value: "Fruit7" },
	// 	<invalid>,
	// 	{ Key: 5, Value: "Fruit5" },
	// 	<invalid>,
	// 	{ Key: 3, Value: "Fruit3" },
	// 	<invalid>,
	// 	{ Key: 1, Value: "Fruit1" }
	// ]

	Shrink 将从容器的末端移除所有slack，但这会在中间或开始处留下空白元素。

	Compact和Shrink一起可以删除所有slack。
	FruitMap.Compact();
	// FruitMap == [
	// 	{ Key: 9, Value: "Fruit9" },
	// 	{ Key: 7, Value: "Fruit7" },
	// 	{ Key: 5, Value: "Fruit5" },
	// 	{ Key: 3, Value: "Fruit3" },
	// 	{ Key: 1, Value: "Fruit1" },
	// 	<invalid>,
	// 	<invalid>,
	// 	<invalid>,
	// 	<invalid>
	// ]
	FruitMap.Shrink();
	// FruitMap == [
	// 	{ Key: 9, Value: "Fruit9" },
	// 	{ Key: 7, Value: "Fruit7" },
	// 	{ Key: 5, Value: "Fruit5" },
	// 	{ Key: 3, Value: "Fruit3" },
	// 	{ Key: 1, Value: "Fruit1" }
	// ]

KeyFuncs
	用到的时候再查头上的在线文档吧。
```
```
TSet
TArray
```
# 委托

# 代码规范


# 其他
## c++ usage

- compile-time checking
```
static_assert(condition, "condition_failed str")
```

- 模板类实例化
```
template<class A, class B>
struct TAreTypesEqual
{
  static const bool Value = false;
};
// 模板类实例化？
template<class T>
structTAreTypesEqual<T, T>
{
  static const bool Value = true;
};
```

- c++ 11 enumerate base or scoped enumeration
```
include <cstdint> // For int32_t

enum Type : int32_t { 
    Value1, // implicitly 0
    Value2, // implicitly 1
    Value3  // implicitly 2
};

int main() {
    Type myValue = Value2;
    
    // The underlying type of myValue is int32_t (a 32-bit integer).
    
    return 0;
}


enum class Type : int32_t {
    Value1,
    Value2,
    Value3
};

```

- RAII
```
RAII, which stands for Resource Acquisition Is Initialization, is a programming idiom used in several object-oriented languages, notably C++. The principle behind RAII is that object lifetime and resource management – such as memory allocation, file handles, socket connections, etc. – are tightly coupled. This means that resource allocation (acquisition) is done during object creation (initialization), and resource deallocation (release) is automatically handled when the object is destroyed. This approach leverages the deterministic object destruction in languages like C++, where destructors are automatically called when objects go out of scope.

class FileHandler {
private:
    std::fstream file;
public:
    FileHandler(const std::string& filename) {
        // Resource acquisition: Opening a file on object creation
        file.open(filename, std::ios::out | std::ios::app);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file.");
        }
    }

    void write(const std::string& message) {
        if (file.is_open()) {
            file << message << "\n";
        }
    }

    ~FileHandler() {
        // Resource release: File is automatically closed when object is destroyed
        file.close();
    }
};

void useFile() {
    FileHandler fh("example.txt");
    fh.write("Hello, RAII!");
    // Once fh goes out of scope, its destructor will automatically close the file
}

int main() {
    useFile();

    return 0;
}
```  

- parameter pack "typename..." for variadic template function
```
// Base case for recursion
void print() {
    std::cout << "End of arguments." << std::endl;
}

// Template function accepting a variable number of arguments
template<typename T, typename... Args>
void print(const T& firstArg, const Args&... args) {
    std::cout << firstArg << std::endl; // Process the first argument
    print(args...); // Call print for the rest of the arguments, peeling off one argument at a time
}

int main() {
    print(1, 2.5, "three", '4');
    return 0;
}
```

- parameter pack for variadic class template
```
template<typename... Values>
class Tuple {};

int main() {
    Tuple<int, double, std::string> myTuple;
    return 0;
}
```

- Remove Reference
```
TRemoveReference is a template utility in Unreal Engine's C++ code base that is designed to strip away the reference qualifier from a given type. This means, if you have a type T& (a reference to T), applying TRemoveReference to it would result in T. Similarly, if the type is T&& (an rvalue reference), it also results in T. If the type T does not have a reference qualifier, it remains unchanged.

要理解这个东西，设计template metaprogramming， type traits and referene collapsing rules.
Metaprogramming with templates allows C++ programmers to write code that operates on other types, introspects types, and transforms types, all at compile time.

Unreal Engine implementation:
/**
 * TRemoveReference<type> will remove any references from a type.
 */
template <typename T> struct TRemoveReference      { typedef T Type; };
template <typename T> struct TRemoveReference<T& > { typedef T Type; };
template <typename T> struct TRemoveReference<T&&> { typedef T Type; };

也可以这么写

template <typename T> struct TRemoveReference      { using Type = T; };
template <typename T> struct TRemoveReference<T& > { using Type = T; };
template <typename T> struct TRemoveReference<T&&> { using Type = T; };
```



- decltype(expression) c++11
- decltype(auto) c++14





## Runtime/Core
```
Cross-Platform
Performance
Modularity
Fundation for Engine Modules
Game Development use higher-level abstractions or explicitly when low-level operations are needed for optimization.
```
### Public
#### Templates
```
ChooseClass.h
	TChooseClass<bool, class1, class2>::Result  // Class type determined during compiling time.
	


TIsSigned<T>::Value
TIsIntegral<T>::Value
TIsPointer<T>::Value
TIsArray::Value                               没看明白？？？
TIsBoundedArray::Value                        没看明白？？？
TIsUnboundedArray::Value                      没看明白？？？
TIsArrayOrRefOfType<>::Value                  没看明白？？？
TIsClass::Value   // Whether is struct/class, 没看明白怎么实现的？？？
...

TMoveSupportTraits<T>::Copy
TMoveSupportTraits<T>::Move

// Get the minimal multiple of alignment that is >= value.
Align<T>(T value, uint64 alignment)           // alignment must be power of 2 
AlignDown<T>(T value, uint64 alignment)       // alignment must be power of 2
AlignArbitrary<T>(T value, uint64 alignment); // arb
TIsAligned<T>(T value, uint64 alignment)


AndOrNot.h: 没看明白。

ARE_TYPES_EQUAL(A,B)

Function.h
	TIsTFunction<xxx>::Value
	TIsTUniqueFunction<xxx>::Value
	TIsTFunctionRef<xxx>::Value
	TFunction<FuncType>
	TUniqueFunction<FuncType>  // 例： TUniqueFunction<void()> CompletionCallback;
	
Identity.h
	TIdentity<T>::Type		
RemoveReference.h
	TRemoveReference

UnrealTemplate.h
	template<typename T>
	FORCEINLINE void Move(T& A,typename TMoveSupportTraits<T>::Copy B)
	{
		// Destruct the previous value of A.
		A.~T();
	
		// Use placement new and a copy constructor so types with const members will work.
		new(&A) T(B);
	}
	TKeyValuePair
	IfAThenAElseB
	IfPThenAElseB
	XOR
	Move
	template <typename T>
	FORCEINLINE typename TRemoveReference<T>::Type&& MoveTemp(T&& Obj)
	{
		typedef typename TRemoveReference<T>::Type CastType;
	
		// Validate that we're not being passed an rvalue or a const object - the former is redundant, the latter is almost certainly a mistake
		static_assert(TIsLValueReferenceType<T>::Value, "MoveTemp called on an rvalue");
		static_assert(!TAreTypesEqual<CastType&, const CastType&>::Value, "MoveTemp called on a const object");
	
		return (CastType&&)Obj;
	}

	template<typename T> T CopyTemp(T& Val){ return Val; }
	template<typename T> T CopyTemp(const T& Val){ return Val; }
	template<typename T> T&& CopyTemp(T&& Val){ return MoveTemp(Val); }

	template<typename T> T&& Forward(typename TRemoveReference<T>::Type& Obj) { return (T&&)Obj; }   // 等价于std::forward, 引用转为rvalue reference，右值引用。 & -> &&
	template<typename T> T&& Forward(typename TRemoveReference<T>::Type&& Obj) { return (T&&)Obj; }

	template <typename T, typename ArgType> T StaticCast(ArgType&& Arg) { return static_cast<T>(Arg); }

	TRValueToLValueReference<T>::Type
	BitMask<uint64>( uint32 Count )
	TForceInitAtBoot

UnrealTypeTraits.h
	TIsDerivedFrom<DerivedType, BaseType>
	/**
	 * TIsFunction
	 *
	 * Tests is a type is a function.
	 */
	template <typename T>
	struct TIsFunction
	{
		enum { Value = false };
	};
	
	template <typename RetType, typename... Params>
	struct TIsFunction<RetType(Params...)>
	{
		enum { Value = true };
	};	

	template<typename T> 
	struct TIsFundamentalType 
	{ 
		enum { Value = TOr<TIsArithmetic<T>, TIsVoidType<T>>::Value };
	};


...
```
#### Internationalization
```
Text.h
  FText

```
#### Containers
```
TArray
TMap
TSet
TArrayView
UnrealString.h
  FString
  TCHAR* GetData(FString&);
  const TCHAR* GetData(const FString&);
  uint32 GetTypeHash(const FString& S)
  FString BytesToString(const uint8& in, count)
  StringToBytes(string, OutBytes, MaxBufferSize)
  ...
  ToCStr
```
#### Algo
```

```
#### Async
异步返回值称为futures

```
Future.h
  FFutureState
    	bool WaitFor(const FTimespan& Duration) const
	{
		if (CompletionEvent->Wait(Duration))
		{
			return true;
		}
		
		return false;
	}
	bool IsComplete() const
	{
		return Complete;
	}
	void SetContinuation(TUniqueFunction<void()>&& Continuation)
	{
		bool bShouldJustRun = IsComplete();
		if (!bShouldJustRun)
		{
			FScopeLock Lock(&Mutex);
			bShouldJustRun = IsComplete();
			if (!bShouldJustRun)
			{
				CompletionCallback = MoveTemp(Continuation);
			}
		}
		if (bShouldJustRun && Continuation)
		{
			Continuation();
		}
	}

  TFutureState<InternalResultType>
  TFuture<ResultType>
  TSharedFuture<ResultType>
  TPromise<ResultType>
  	
```
#### Serialization
```
FArchiveState
FArchive
```
#### Memory Management
#### Threading/Concurrency
#### Math Library
#### Reflection and Metadata
#### Event System
#### Internationalization and Localization
#### Configuration System
#### Misc
```
AES.h
  AES加密 FAES::EncryptData(uint8& content, numbytes, const FAESKey& Key);

App.h
  Application related static functions, FApp::GetBranchName()

AsciiSet.h
  FAsciiSet::HasAny(str, FAsciiSet set)

AssertionMacros.h
  FDebug
    FDebug::DumpStackTraceToLog(LogVerbosity)
  _DebugBreakAndPromptForRemote()
  checkCode( Code );
  check(expr)
  verify(expr)
  checkf(expr, format, ...)
  verifyf(expr,format,...)
  checkNoEntry()
  checkNoReentry()
  checkNoRecursion()
  checkSlow(expr)
  checkfSlow(expr, format, ...)
  verifySlow(expr)
  LowLevelFatalError(Format, ...)
  
AutomationTest.h
  FAutomationTestInfo
  FAutomationExecutionInfo
  IAutomationLatentCommand // Time deffered test
  FThreadedAutomationLatentCommand //
    if (!Future.IsValid())
		{
			Future = Async(EAsyncExecution::Thread, Function);
		}

		return Future.IsReady();
  IMPLEMENT_SIMPLE_AUTOMATION_TEST



Build.h
  #if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
  #endif


CoreDefines.h
  STUBBED(str) // 存根，打印一次就不再打印
CoreMiscDefines.h
  PURE_VIRTUAL(func,...) { LowLevelFatalError(TEXT("Pure virtual not implemented (%s)"), TEXT(#func)); __VA_ARGS__ }
  UE_NONCOPYABLE(TypeName)
  UE_DEPRECATED(Version, Message)

DateTime.h
  EDayOfWeek::Monday
  EMonthOfYear::Jannuary = -1
  
  FDataTime

Guid.h
	FGuid
Paths.h
  FPaths::EngineDir()
  FPaths::IsSamePath
  FPaths::MakePathRelativeTo(A, B)
  FPaths::Split()

Parser.h
  FParse::Value(Stream, Match, FName& Nalue);


ScopeLock.h
  FScopeLock xxx()
ScopeRWLock.h
  FReadScopeLock
  FWriteScopeLock
  FRWScopeLock

TimeSpan.h
  ETimespan::MaxTicks, ...
  FTimeSpan
    FTimeSpan::FromDays()
    FTimeSpan::FromHours()
    FTimeSpan::Parse(FString TimespanString, FTimeSpan& OutTimeSpan);

OutputDevice.h
  SET_WARN_COLOR(Color)
  SET_WARN_COLOR_AND_BACKGROUND(Color, Bkgrnd)
  CLEAR_WARN_COLOR(Color)
  FOutputDevice

```


#### Logging
Logging/LogMacros.h
```
Log Category:
  LogTemp for temporary debugging.
  LogPhysics for physics-related messages.
  Custom category

Verbosity Levels:
  Log
  Warning
  Error

Macro for Logging:
  UE_LOG(LogCategory, Verbosity, FormatString, ...)
  UE_WARN
  UE_ERROR

Using the logging system:
  DECLARE_LOG_CATEGORY_EXTERN(LogCategory, DefaultVerbosity, CompilationTimeVerbosity)
  DEFINE_LOG_CATEGORY(LogCategory)

Features:
  File Output: Saved/Logs
  Real-Time Filtering: 
  Color-Coded:
  Screen Logging:  GEngine->AddOnScreenDebugMessage()

Configuration:
  Config Files: DefaultEngine.ini
  Command Line:

Best Practices:
```
#### HAL
```
Event.h
  FEvent
  FEventRef
    // RAII
    FEventRef::FEventRef(EEventMode Mode /* = EEventMode::AutoReset */)
    	: Event(FPlatformProcess::GetSynchEventFromPool(Mode == EEventMode::ManualReset))
    {}
    FEventRef::~FEventRef()
    {
    	FPlatformProcess::ReturnSynchEventToPool(Event);
    }


Platform.h
  

```
#### GenericPlatform
```
GenericPlatformProcess.h
  TProcHandle<T, T InvalidHandeValue>
  FProcHandle
  FGenericPlatformProcess
    FEvent* GetSynchEventFromPool(bool bIsManualReset = false);
    void FlushPoolSyncEvents();
    void ReturnSynchEventToPool(FEvent* Event);

```
#### Windows
```
WindowCriticalSection.h
  FWindowsRWLock  typedef to FRWLock
  FWindowsCriticalSection typedef to FCriticalSection
  FWindowsSystemWideCriticalSection typedef to FSystemWideCriticalSection

WindowsPlatformTime.h
  FPlatformTime::Seconds() // 返回double
  FPlatformTime::Cycles() // 返回int32
  FPlatformTime::Cycles64() // 返回int64
  FPlatformTime::SystemTime(Year, Month, DayOfWeek, Day, Hour, Min, Sec, MSec);
  FPlatformTime::UtcTime(Year, Month, DayOfWeek, Day, Hour, Min, Sec, MSec);
  FPlatformTime::UpdateCPUTime(float DeltaTime);

WindowsPlatformProcess.h
  FPlatformProcess
    
    GetSynchEventFromPool(true)

    Sleep(Seconds)
    GetDllHandle(const TCHAR* Filename)
    uint32 GetCurrentProcessId();
    uint32 GetCurrentCoreNumber();
    ...
    FExecutablePath()
    WaitForProc
    GetApplicationMemoryUsage
    Sleep
    FEvent* CreateSynchEvent(bool bIsManualReset = false);
    FRunnableThread* CreateRunnableThread()
    CreatePipe(...)
    ClosePipe()
    WritePipe
    ReadPipe
    FProcHandle CreateProc( const TCHAR* URL, const TCHAR* Parms, bool bLaunchDetached, bool bLaunchHidden, bool bLaunchReallyHidden, uint32* OutProcessID, int32 PriorityModifier, const TCHAR* OptionalWorkingDirectory, void* PipeWriteChild, void * PipeReadChild = nullptr);
    SetCurrentWorkingDirectoryToBaseDir()
    FString GetCurrentWorkingDirectory()
```




## Runtime/CoreUObject
CoreUObject.h
### Public
#### AssetRegistery
#### Blueprint
#### Internationalization
#### Misc
#### Serialization
#### Templates
#### UObject
```
Object.h    // UObject is in here
Field.h     // Base class for reflection data object.
	


```


## Runtime/Json
是对nlohmann的封装，直接把源代码放进插件中了。
### Public
Json.h
JsonGlobals.h
```
目录 Serialization, Dom, Policies
EJsonXXX
TJsonWriter
TJsonReader
FJsonStringReader
TJsonReaderFactory
FJsonSerializerWriter
FJsonSerializer
```



## Runtime/Engine
### Classes
#### Engine
Engine.h
### Public


## ThirdParty
c#写法添加include/lib

