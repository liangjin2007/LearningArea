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

# 组件
```
组件 是一种特殊类型的 对象，Actor 可以将组件作为子对象附加到自身。组件适用于共享相同的行为，例如显示视觉表现、播放声音。它们还可以表示项目特有的概念，例如载具解译输入和改变其速度与方向的方式。
举例而言，某个项目拥有用户可控制车辆、飞机和船只。可以通过更改载具Actor所使用的组件来实现载具控制和移动的差异。
```
- Actor组件 ActorComponent
```
UActorComponent 是所有组件的基类。由于组件是渲染网格体和图像、实现碰撞和播放音频的唯一方法，因此玩家游戏期间在场景中看到或进行交互的一切其实都是某一类组件的成果。
注册组件
	注册事件
取消注册组件
	取消注册事件
更新
渲染状态
物理状态
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

