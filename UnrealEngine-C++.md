# [用C++编程](https://dev.epicgames.com/documentation/en-us/unreal-engine/programming-with-cplusplus-in-unreal-engine) 
- 概述
- Actor
- [组件Components]()
- 反射系统
- 游戏性架构
- 虚幻引擎中的容器
- 委托
- 代码规范
- 其他


# 概述
# AActor
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



# 组件
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
视觉化组件
```  
- 场景组件 SceneComponent
```
附加
```
- 基元组件 PrimitiveComponent
```
场景代理
```  








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

