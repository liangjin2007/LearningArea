# Unreal Engine
参考UE4 C++进阶之路，

- 入门指南视频: https://learn.unrealengine.com/course/3584597/module/6933184?moduletoken=UHxxnDLPW8TDVsMz2Aaiq0E499zOCpVZy3JR-t10g1IA4WGsvwP4wGnV-ug1GsJ7&LPId=114305
- P1 https://www.bilibili.com/video/BV1C7411F7RF?p=1
- P2 https://www.bilibili.com/video/BV1C7411F7RF?p=2
- bilibili搜索"虚幻4游戏引擎官方入门自学到提高全218集" https://www.bilibili.com/video/BV12s411P7PW?from=search&seid=17955912259534568723
- 新建插件https://docs.unrealengine.com/zh-CN/Programming/Plugins/index.html
- 插件 Boost_PCL_UnrealThirdPartyPlugin
- [bilibili虚幻引擎官方](https://space.bilibili.com/138827797?spm_id_from=333.788.b_765f7570696e666f.2)
- [虚幻引擎中文技术直播 第1期 虚幻引擎4的实时渲染流程](https://www.bilibili.com/video/BV1yb411c7in)
- [UE4 Data Driven Development](https://www.bilibili.com/video/BV1dk4y1r752)
- Unrela engine 4官方教程1-98 带中文字幕 https://www.bilibili.com/video/BV1vJ411J7eF?p=1
- 最新添加 UnrealEngine官方程序范式分析.pdf https://max.book118.com/html/2019/0327/7053133100002015.shtm
- 以及书中提到的《Inside UE4》https://zhuanlan.zhihu.com/p/22813908
- UE5.4文档 https://dev.epicgames.com/documentation/zh-cn/unreal-engine/unreal-engine-5-4-documentation

## UE 5.4文档


- 13 shading models https://www.youtube.com/watch?v=-mAcsaMDuaw
- 9 input modifiers https://www.youtube.com/watch?v=MN-0otRNmZI
- About Shading Models https://advances.realtimerendering.com/s2023/2023%20Siggraph%20-%20Substrate.pdf
  




## 概念
```
Unit : 默认是cm
Camera Speed
Sphere Reflection Actor
PostProcessVolume
SkyLight
SkySphere
DirectionLight
Event
Event Dispatcher
UMG
Deferred/Forward rendering
Sound Cue
Custom Depth/Stencil用来在实时渲染中进行合成
```





-  [渲染](https://www.bilibili.com/video/BV1yb411c7in)
```
Unreal 实时渲染流程：
剔除-> Draw Call合并属性等-> 光栅化-> Early z pass:Depth Buffer避免重复绘制Pixel-> Custom Depth/Stencil -> GBuffers -> Static Light -> Dynamic Light -> Stationary Light -> 反射Reflection -> Translucency Sortt -> Post Processing

其中GBuffers
  Base Color
  Normal
  Metalic
  Roughness
  Specular
  SubSurface
  Shading Model
  Depth


100个物体+100个灯光
延迟渲染： 减少绘制次数 100 + 100， 以及避免像素重复计算
前向渲染: 100 x 100

静态光：
light map : 用lightmass程序将颜色和阴影提前做出来算出lightmap, 绘制的时候只要调这个图片就行。
  离线
  Bake时间
  Rebuild
  消耗内存
  漂亮的间接光
  反弹和GI
  
动态光：
  使用G-Buffer
  全实时
  Heavy
  没有间接光
  
Stationary Light
  间接光效果可以通过volumetric lightmap计算。
  将间接光和反弹细节存入这些点

反射
  SSR： Screen Space Reflection
    系统默认反射方式
    实时，精确
    只能反射屏幕上有的物体
  Reflection Captures
    Cubemap
    预先计算Cubemap
    快，省
    离Capture越远越不精确
  Planar Reflection
    实时反射全环境
    平面：镜子，水面
    消耗大
    
DOF of Translucency
  透明物体不绘制Depth Buffer
  使用Separate Translucent Buffer
  DOF function在shader中处理

Post Processing 
  有固定路径： DOF > Motion Blur > Bloom > Tone mapping
```



- [放置物体](https://www.bilibili.com/video/BV1vJ411J7eF?p=5)
```
1. 从放置actor拖进视口
2. 从类查看器直接拖进视口 ( 窗口 -> 开发者工具 -> 类查看器(class viewer) )
3. 从内容浏览器拖进视口
4. 在视口中explore到某个地方通过右键可选择最近放置过的对象进行放置。
5. TODO: 其他方式
```

- [内容浏览器](https://www.bilibili.com/video/BV1vJ411J7eF?p=10)
```
导入功能
Place功能：比如直接拖放Actor， 材质到Viewport。
双击可以编辑从而打开各种不同类型的编辑器： 最常用的是静态网格体。 打开界面后可以查看uv, 法向，包围体，碰撞体等信息。
最多可以开四个内容浏览器窗口
锁定
从详细界面点击一个搜索图标跳到内容浏览器
过滤器 Filters
视图选项 view options
  列显示，瓦片显示，etc
  是否显示文件夹
  显示引擎内容
  显示开发者内容
  缩略图编辑模式
```


- [设计 Design](https://www.bilibili.com/video/BV1vJ411J7eF?p=12)
```
分几个阶段
  Prototype Pass :  用简单的mesh or geometry先搭一个原型出来
  Meshing Pass : 简单光照，接近完成的网格/材质资产
  Lighting Pass : 放置光源，调整材质，调整后处理效果
  Polish Pass : 添加效果，Reflection Actors, Blocking Volumes, Audio, more details
```

- [关卡创建](https://www.bilibili.com/video/BV1vJ411J7eF?p=13)
```
1. 放置actor中“基础”与“几何体”是不同的。 几何体中是Brush对象，可选择Additive或者Subtractive
2. 关卡创建中使用的是几何体。
3. 选中一个Actor后可右键->控制或者按快捷键Ctrl+Shift+P切换到使用当前actor来飞行的模式，可在Viewport左上角透视下方关闭。
4. 关于玻璃：  按T键 可以决定鼠标点击时，玻璃物体是否会被选中。 或者界面上在 设置 -> 允许选择半透明可切换。类似地有选择组的功能，快捷键是Ctrl+Shift+G。
5. 蓝图推门 ： 放置actor -> 基础 -> 



5. 问题： 如何在已有Actor上开个洞？ 
```

- [蓝图](https://www.bilibili.com/video/BV1vJ411J7eF?p=24)
```
what
where
how


1. 用蓝图控制开关灯




```

## 开发经验
```
一、编译源码
1. .\Setup.bat
2. .\GenerateProjectFiles.bat 2017
3. 打开VS编译UE4 project， 其他不用编译
4. 编译UE4源码时，不能开着UE Editor。

二、基于源码的编程方式
F5启动UE4工程会启动UE4 Editor。如果启动不了，需要设置UE4项目的Debugging属性
Command: $(TargetPath)
Working Directory: $(ProjectDir)
Attach:No
Debugger Type ： Auto
Merge Environment : Yes
SQL Debugging ： No

三、**新建的UE项目千万不能重编译，否则会重编译UE引擎**。

四、UE Editor中需要设置源代码托管。 如果直接从P4或者git更新代码，有时会失败。

五、蓝图-->类设置 可以查看BP的父类信息。

六、调试UE项目
需要项目属性配置成二中一样。特别要注意不要将Attach设成true。 
F5调试时会自动启动UE Editor。

七、日志
UE_LOG(LogTemp, Warning, TEXT("%s"), *(FString));

八、头文件
必须放在.generated.h之前

九、学UE先学蓝图，然后再学C++。
 
十、C++其实是UE的脚本语言，学习难度是蓝图的2倍不止
 
十一、新建C++类就可以把蓝图工程转成C++工程

十二、BP:可以理解为是一个editor

十三、如何设置引擎的语言为英文/中文
```

- awesome-ue4 https://github.com/terrehbyte/awesome-ue4

## 界面和蓝图相关


#### 交互和快捷键
- 快捷键
```
Ctrl+~: 切换世界空间和对象空间： 在视口右上角。 方便沿着某个方向移动。
Ctrl+Shift+F : 打开内容浏览器
F11: switch emerce mode
Ctrl+number: label camera
F: focus to something
Ctrl+Shift+Save
Alt+drag/rotate/scale in viewport: copy 
Ctrl+G: 分组
Ctrl+Shift+G : 取消分组
alt+shift+r查找引用
alt+shift+f查找变量引用
End: 落地键
ctrl+B跳到资源
ctrl+w复制一份（先选中）
E： 旋转
D： Delay
按住ctrl, 把variable 拖到bp event graph编辑器中
材质蓝图中：
 T键： Texture Sample
 Multiply节点
 L键：Lerp节点
 V键：vector节点
 scalar parameter可以被代码修改
```



- 交互
```
Layers: Double Click to select
Select actors using this asseset.
UMG Layout， 内置控件，创建UI， 图表（Graph）, Input Mode, Input Action, 自定义(Custom)事件及调用， 基础游戏循环逻辑，十字星，血条，
生命值和弹药UI。
右键Wrap with xxx
在已有UMG上右键点Duplicate
Event定义在Character蓝图，这个时候Event的Target变量是Character蓝图类型。 
PlayController蓝图里获取Character蓝图对象， 设为Event的Target。
Input Action Pause的细节面板中有Execute when Paused
搜索类，搜索BP, 搜索选项。
动作映射，用什么键控制什么动作， 动作需要先添加一下。Project Settings -> Inputs -> Action 
如何让BP中定义的Component变量在Actor的Default中显示变量
- Word Settings -> 设置GameMode override为自定义GameMode
视口 -> Lit-> Buffer Display 光照->缓冲显示
视口 -> Show -> Visualize -> Volumetric Lightmap
```


- 声音
```
添加wav文件到content browser
右键create cur船舰Sound Cue
查看Light maps： World Settings->Lightmaps
```



#### API
```
变量： 变量类型有Bool, Text/String, Int, Float， etc。还有结构体，枚举，对象类型，接口，变换，向量，旋转等类型。
节点： 
  节点分类： 执行节点，读取节点，事件节点
  节点注释：注释组，注释框
  情境关联
  
Show Mouse Cursor
Open Level
Quit Game
Create Widget节点
Add To Viewport节点
删除UI: Remove From Parent
Set variable to nothing
Set Game Paused节点
Open Level
Set Input Mode Input节点
Set Input Mode Game And UI
Get Player Controler节点
Set Show Mouse Cursor节点
Get Show Mouse Cursor节点
判断蓝图对象是否已经创建 
  节点的Return Value -->  拖出去 --> Promote to variable
  Get xxx --> Is Valid节点
Make Literal xxx
AddImpose
Tick中deltaseconds的用法
Print String
LineTraceByChannel
 Draw Debug Type设为Persistent，可以把这条射线画出来
 Trace Channel跟Actor的Collision相关
GetActorLocation
Break Hit Result
  Impact Normal
EventActorBeginOverlap
Cast to
Set Physics Linear Velocity
Timeline
SetRelativeLocation
Set Parameter Value on Material xxx
Set xxx
Spawn Sound Attached
Construction Script
Select
Moved Component To
Get Actors of Class
ForEachLoop
Switch On xxx
Add 把元素添加到数组类variable中
DestroyActor
Bind Event to OnDestroyed
Delay
Set Volume Multiplier设置音量
Set Pitch Multiplier设置音调
Get Velocity
Vector Length
+，-，*，/, %除以
Clamp
Add Static Mesh Component
Promote to Variable存成变量
Construction Script
 SetRelativeLocation
 GetRelativeLocation
 Find look at Location
GameMode统筹整个游戏的游戏数据
P
```
#### 经验/命令
```
命令
r.xxx command http://www.kosmokleaner.de/ownsoft/UE4CVarBrowser.html
```


## C++相关
#### 基础
```
消息
钩子
用C++将属性公开给蓝图
通过蓝图扩展C++类
蓝图调用C++函数
C++函数调用蓝图中定义的函数
蓝图VM
虚实调用
反射
垃圾回收
咒语
 UCLASS(config=Game)
 GENERATED_BODY()
 UPROPERTY
 UFUNCTION(BlueprintCallable, Category="Damage") void CalculateValues(); 让蓝图调用C++函数
 负责处理将C++函数公开给反射系统， BlueprintCallable 选项将其公开给蓝图虚拟机
 UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = Camera, meta = (AllowPrivateAccess = "true")) class USpringArmComponent* CameraBoom;
```
#### 类实例
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
UAnimBlueprint
```


#### 其他经验
- [集成opencv](https://nerivec.github.io/old-ue4-wiki/pages/integrating-opencv-into-unreal-engine-4.html)


## 应用
https://github.com/bw2012/UnrealSandboxTerrain

https://github.com/20tab/UnrealEnginePython

https://github.com/facebookarchive/UETorch  

https://github.com/adynathos/AugmentedUnreality

https://github.com/tomlooman/EpicSurvivalGameSeries

https://github.com/iniside/ActionRPGGame

https://github.com/getnamo/tensorflow-ue4

https://github.com/microsoft/AirSim

https://github.com/ValentinKraft/Boost_PCL_UnrealThirdPartyPlugin

https://github.com/tomlooman/ActionRoguelike


Body Related:
https://paperswithcode.com/task/3d-reconstruction

https://paperswithcode.com/task/3d-shape-reconstruction

https://paperswithcode.com/task/pose-estimation

https://paperswithcode.com/task/3d-pose-estimation

https://paperswithcode.com/task/monocular-3d-human-pose-estimation

https://paperswithcode.com/task/3d-absolute-human-pose-estimation

https://paperswithcode.com/task/3d-multi-person-pose-estimation-absolute

metahuman https://metahuman.unrealengine.com/


## 实践

- 清理C盘
```
清理C盘重点目标位置
1. Local Temp C:\Users\liang.jin\AppData\Local\Temp 大概能清个10几个G出来。
2. VSCode相关 C:\Users\liang.jin\AppData\Roaming\Code 大概能清歌20个G出来
	User
	Cache
	CachedData
3. conda C:\Users\liang.jin\.conda 大概几个G的空间。
4. 安装megascan的程序Bridge后 C:\Users\liang.jin\Documents\Megascans Library 此目录大概有几百兆。
5. C:\Users\liang.jin\AppData\Local\pip 几个G
```

- neural-blend-shapes
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
