# UE C++ API
- UE4中的C++编程简介 https://docs.unrealengine.com/zh-CN/Programming/Introduction/index.html

- 框架是创建**Gameplay系统**
  - ?? 包括哪些知识点呢？
  - 消息？ ...
  - 钩子?
  
- 用C++将属性公开给蓝图
- 通过蓝图扩展C++类
- 蓝图调用C++函数
  - UFunction
- C++函数调用蓝图中定义的函数
- 蓝图VM
- 虚实调用
- UE4 Data Driven Development https://www.bilibili.com/video/BV1dk4y1r752
- 反射
- 垃圾回收
- 咒语
  - UCLASS(config=Game)
  - GENERATED_BODY()
  - UPROPERTY
  - UFUNCTION(BlueprintCallable, Category="Damage") void CalculateValues(); 让蓝图调用C++函数
    - 负责处理将C++函数公开给反射系统， BlueprintCallable 选项将其公开给蓝图虚拟机
  ```
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = Camera, meta = (AllowPrivateAccess = "true"))
	class USpringArmComponent* CameraBoom;
  ```
- 在线课程
https://www.bilibili.com/video/BV1fE411a74g/?spm_id_from=333.788.videocard.3

- 背景音乐

- 课程PPT

- RBF https://docs.unrealengine.com/en-US/API/Runtime/AnimGraphRuntime/RBF/index.html
  - Entity，Target, Radius, Scale, Normalization, MedianReference
  - 半径如何理解？

- Normalize Weight Method
- Weight Threshold to remove contribution



```
enum class ERBFNormalizeMethod : uint8
{
	/** Only normalize above one */
	OnlyNormalizeAboveOne,

	/** 
		Always normalize. 
		Zero distribution weights stay zero.
	*/
	AlwaysNormalize,

	/** 
		Normalize only within reference median. The median
		is a cone with a minimum and maximum angle within
		which the value will be interpolated between 
		non-normalized and normalized. This helps to define
		the volume in which normalization is always required.
	*/
	NormalizeWithinMedian
};
```
## 类实例
```
一般类型
TArray<float>
TMap<FName, FPoseDriverTarget>
TStatId
int32

FRBFOutputWeight
ERBFDistanceMethod // Enum
EPoseDriverSource
EObjectFlags
EObjectMark

FName
FString
FVector
FRotator: Euler角， roll, pitch, yaw
FQuat
FQuat(FRotator)
FTransform
FPoseLink
FBoneReference
FBoneContainer
FPoseDriverTransform
FPoseDriverTarget
FAnimNode_PoseHandler
FAnimNode_PoseDriver
FRBFParams
FRichCurve
FPlatformAtomics
FFieldCompiledInInfo

宏
USTRUCT()
UCLASS()
GENERATED_BODY()
UPROPERTY(EditAnywhere, Category=RBFData)
FORCENOINLINE

枚举
UENUM()
enum EBoneAxis
{
	BA_X UMETA(DisplayName = "X Axis"),
	BA_Y UMETA(DisplayName = "Y Axis"),
	BA_Z UMETA(DisplayName = "Z Axis"),
};
UENUM(BlueprintType)
enum EBoneControlSpace
{
	...
}

U类型，表示UObject的继承者，是一种对象。 UObject_Base -> 
UObjectBase
  - GetClass
  - GetOuter
  - GetFName
  - GetUniqueID
  - Register()
  - DeferredRegister()
  - AddObject()
  - GetStatID()
  - SetFlagsTo( EObjectFlags NewFlags )
  - GetFlags()
UObjectBaseUtility
  - SetFlags( EObjectFlags NewFlags )
  - HasAnyFlags
  - Mark(EObjectMark Marks) const
  - GetFullName()
  - GetPathName()
  - CreateClusterFromObject
  - GetFullGroupName
  - GetName
  - IsA
  - FindNearestCommonBaseClass
  - GetLinker
USkeleton
USkeletalMesh
UPoseAsset

通用写法
Node.xxx
例子
Node.GetRBFTargets(RBFTargets);

```
