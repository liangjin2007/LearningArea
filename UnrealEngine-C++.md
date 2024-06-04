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

UnrealTemplate.h
  IfAThenAElseB
  IfPThenAElseB
  XOR
  Move


// Get the minimal multiple of alignment that is >= value.
Align<T>(T value, uint64 alignment)           // alignment must be power of 2 
AlignDown<T>(T value, uint64 alignment)       // alignment must be power of 2
AlignArbitrary<T>(T value, uint64 alignment); // arb
TIsAligned<T>(T value, uint64 alignment)


AndOrNot.h: 没看明白。

ARE_TYPES_EQUAL(A,B)

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

  TFutureState<InternalResultType>
  
```
#### Serialization
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





## Runtime/Engine
### Classes
#### Engine
Engine.h
### Public


## ThirdParty
c#写法添加include/lib

