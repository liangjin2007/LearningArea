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

// Get the minimal multiple of alignment that is >= value.
Align<T>(T value, uint64 alignment)           // alignment must be power of 2 
AlignDown<T>(T value, uint64 alignment)       // alignment must be power of 2
AlignArbitrary<T>(T value, uint64 alignment); // arb
TIsAligned<T>(T value, uint64 alignment)


AndOrNot.h: 没看明白。

```
#### Containers
```
TArray
TMap
TSet
TArrayView
```
#### Algo
```

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
宏
  CoreDefines.h
    STUBBED(str) // 存根，打印一次就不再打印
  CoreMiscDefines.h
    PURE_VIRTUAL(func,...) { LowLevelFatalError(TEXT("Pure virtual not implemented (%s)"), TEXT(#func)); __VA_ARGS__ }
    UE_NONCOPYABLE(TypeName)
    UE_DEPRECATED(Version, Message)
    
HAL/Platform.h
  int8, int16, .., uint8, uint16, .., ANSICHAR, WIDECHAR, TCHAR, UTF8CHAR, UCS2CHAR, UTF16CHAR, UTF32CHAR


AssertionMacros.h
```
Containers/Array.h
Containers/Map.h


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


## Runtime/Engine
### Classes
#### Engine
Engine.h
### Public


## ThirdParty
c#写法添加include/lib

