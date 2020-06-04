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




