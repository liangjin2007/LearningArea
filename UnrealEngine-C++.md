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
### Public
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

