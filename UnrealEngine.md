# Unreal Engine
参考UE4 C++进阶之路，

- 入门指南视频: https://learn.unrealengine.com/course/3584597/module/6933184?moduletoken=UHxxnDLPW8TDVsMz2Aaiq0E499zOCpVZy3JR-t10g1IA4WGsvwP4wGnV-ug1GsJ7&LPId=114305
- P1 https://www.bilibili.com/video/BV1C7411F7RF?p=1
- P2 https://www.bilibili.com/video/BV1C7411F7RF?p=2
- bilibili搜索"虚幻4游戏引擎官方入门自学到提高全218集" https://www.bilibili.com/video/BV12s411P7PW?from=search&seid=17955912259534568723
- 新建插件https://docs.unrealengine.com/zh-CN/Programming/Plugins/index.html
- 泰课网 https://www.taikr.com/course/964/task/27889/show
- 插件 Boost_PCL_UnrealThirdPartyPlugin

## 学习
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


学UE先学蓝图，然后再学C++。
 
C++其实是UE的脚本语言，学习难度是蓝图的2倍不止
 
新建C++类就可以把蓝图工程转成C++工程

BP:可以理解为是一个editor

快捷键
F11: switch emerce mode
Ctrl+number: label camera
F: focus to something
Ctrl+Shift+Save
Alt+drag/rotate/scale in viewport: copy 
Ctrl+G: 分组
Ctrl+Shift+G : 取消分组

Camera Speed
Sphere Reflection Actor
PostProcessVolume
SkyLight
SkySphere
DirectionLight
Layers: Double Click to select
Select actors using this asseset.

UI相关
UMG Layout， 内置控件，创建UI， 图表（Graph）, Input Mode, Input Action, Show Mouse Cursor， Open Level, Quit Game， 自定义(Custom)事件及调用， 基础游戏循环逻辑，十字星，血条，
生命值和弹药UI。

Create Widget节点
Add To Viewport节点
右键Wrap with xxx
删除UI: Remove From Parent
Set variable to nothing
在已有UMG上右键点Duplicate

Event
Event定义在Character蓝图，这个时候Event的Target变量是Character蓝图类型。 
PlayController蓝图里获取Character蓝图对象， 设为Event的Target。


Input Action Pause的细节面板中有Execute when Paused


游戏相关
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
  
Event Dispatcher

````

```
渲染相关 https://learn.unrealengine.com/home/LearningPath/94219?r=False&ts=637367569407342514

Distance Culling
A mix of different solutions
  把能预计算的都先预先计算好
  然后针对不同的需求用预计算好的数据来实现实时rendering。
  剔除有四种方法

Scalability
  可选择图形画面质量
  允许选择不同级别的内容来适配不同的设备
  r command
  
Deferred/Forward rendering
GBuffer
```
功能
============================================
https://www.unrealengine.com/zh-CN/features
- 管道集成
  - FBX, USD and Alembic支持
  - Python脚本
  - Visual Dataprep
  - Datasmith: 无缝数据转换
  - LiDAR点云支持
  - ShotGun集成
- 世界场景构建
  - 虚幻编辑器
    - 多人编辑
    - VR模式所见即所得
  - 可伸缩的植被
    - 草地工具
  - 资源优化
    - 自动LOD
    - 代理几何体工具
  - 网格体编辑工具
    - 静态网格体编辑器
  - 地形和地貌工具
    - 地形系统
    
- 动画
  - 角色动画工具
    - 网格体编辑工具
    - 动画编辑工具：状态机，混合空间，正向和逆向运动学，物理驱动的布娃娃效果动画，同步预览动画
    - 可编制脚本的骨架绑定系统
  - 动画蓝图
  - LiveLink数据流送
  - Take Recorder
  - Sequencer：专业动画
- 渲染、光照和材质
  - 前向渲染
  - 灵活的材质编辑器
  - 实时进行逼真的光栅化和光线追踪
  - 精细光照：大气层太阳，天空环境，体积雾，体积光照贴图，预计算光照贴图，网格体距离场
  - 色彩准确的最终输出
  - 高品质多媒体输出
  - 先进的着色模型：透明图层，次表面散射，皮肤，毛发，双面植被，薄透明
- 模拟和效果
  - Nigara粒子和视觉效果:火焰，烟雾，尘土和流水。粒子间通信创建连锁式效果，使得例子对音频波形做出反应。
  - 布料工具： Chaos物理解算器；Paint Cloth Tool
  - Chaos物理和破坏系统： 
  - 基于发丝的毛发 
- 游戏性和交互性编写
  - 稳健的多人框架
  - 先进的人工智能
  - UMG
  - 变体管理器
  - 蓝图可视化脚本编制系统
- 集成的媒体支持
  - 专业的视频I/O支持和播放
  - 虚幻音频引擎
  - 媒体框架
- 平台支持和便捷工具
  - 多平台开发
  - VR、AR和MR(XR)支持
  - 像素流送
  - 远程控制协议
  - 搞高效的多屏渲染： nDisplay
  - 面向电影人的虚拟探查
  - 虚拟摄像机插件
- 内容
  - Quixel Megascans
  - 行业特定模板
  - 商业生态圈：虚幻商城
    - 环境，角色，动画，纹理，道具，声音及视觉效果，音轨，蓝图，中间件集成插件，辅助工具以及完整的初学者内容包。
  - 示例项目
- 开发者工具
  - 完全访问C++源代码
  - 无缝集成P4
  - C++ API
  - 分析和性能
  
游戏问题
============================================
- 新建关卡

- 新建场景

- 骨架网格体动画系统 https://docs.unrealengine.com/zh-CN/Engine/Animation/index.html
  - 一些基本功能
    - 多个动画工具和编辑器
    - 结合基于骨架的变形和基于顶点的变形相结合
    - 播放和混合动画序列
    - 创建自定义特殊动作
    - 动画蒙太奇
    - 通过变形目标应用伤害效果或面部表情
    - 使用骨架控制直接控制骨骼变形
    - 创建基于逻辑的状态机来确定角色在指定情境下应该使用哪个动画。
  - 建动画资源 by UE提供的MayaRiggingTool https://docs.unrealengine.com/zh-CN/Engine/Content/Tools/MayaRiggingTool/index.html
    - 跳过，用现有资源走下一步
  - 骨架编辑器 管理驱动骨架网格体和动画的骨骼
  - 骨架网格体编辑器 修改连接到骨架的骨架网格体
  - 动画编辑器 创建并修改动画资源
  - 动画蓝图编辑器 用于创建逻辑、驱动角色使用的动画及使用机制以及动画混合的方式
  - 物理资源编辑器 创建和编辑用于骨架骨架王个体碰撞的物理形体
-  动画系统概述 https://docs.unrealengine.com/zh-CN/Engine/Animation/Overview/index.html
  - ![](https://github.com/liangjin2007/data_liangjin/blob/master/animation.JPG?raw=true)


- 如何设置基本角色为自己提供的角色 https://docs.unrealengine.com/zh-CN/Engine/Animation/CharacterSetupOverview/index.html

- 动画蓝图

- 添加角色动画
https://docs.unrealengine.com/zh-CN/Programming/Tutorials/FirstPersonShooter/4/index.html

[API](http://api.unrealengine.com/latest/CHN/GettingStarted/FromUnity/index.html)

示例与教学
============================================
- 游戏性概念示例
多人连线枪战游戏，虚幻二维任务线条图，蓝图样条曲线轨迹，基于回合的策略游戏，具有UMG的库存UI
- 内容示例
  - 动画内容
  - 音频内容
  - 蓝图内容
  - 贴花
  - 可破坏物内容示例
  - 动态场景阴影
  - 效果内容示例
  - 使用内容示例
  - 导入选项内容示例
  - 地形内容示例
  - 关卡设计内容示例
  - 光照内容示例
  - 材质内容示例
  - 数学运算大厅内容示例
  - 顶点变形对象内容示例
  - 鼠标接口
  - 寻路网格体内容示例
  - 联网内容示例
  - 纸张2D内容示例
  - 物理内容示例
  - Pivot Painter内容示例
  - 后期处理内容示例
  - 反射内容示例
  - 静态网格体内容示例
  - 体积内容示例
  - Matinee内容范例
- 游戏示例项目
- 引擎特性示例
- 虚幻编辑器界面
  - 类查看器 
    - 入口
    - 搜索栏，菜单栏
    - 情境菜单
    - 拖放到视口
    - 类选择器
  - 取色器
    - 颜色空间sRGB
  - 曲线编辑器
    - Distribution
  - 编辑器偏好设置
  - 全局资源选取器
    - Ctrl+P, 然后搜索
    - 拖动资源到场景或双击资源调出相关到编辑器
  - 布局自定义
    - 窗口-> 保存布局
  - 快捷键编辑器
  - 模型预览场景
  - 项目设置
    - 引擎 
    AI系统， 动画， 声音， 碰撞， 控制台， Cooker， Crowd管理， 垃圾回收，通用设置， 输入， 寻路网格， 网格， 物理， 渲染， Streaming数据包读取流，教程，用户界面
    - 编辑器
    Sequencer
    Paper2D
  - 属性矩阵
    - Detail面板
  - 工具和系统
    - 工具
      - 内容制作工具
        - 放置模式
        - 几何体画刷Actor
        - Landscape室外地形
        - 植物工具
        - 静态网格体编辑器UI
        - 动画编辑器
        - 物理资源编辑器
        - 材质编辑器
        - 级联例子编辑器
      - 工具代码编写
      - 场景关卡编辑工具
    - 系统
      - 蓝图-可视化脚本
      -Matinee和过场动画    
  - 源码管理
  - 关卡
    - 关卡分段
  - Actor和几何体
  - 体积参考
    - 物理体积参数
      - 末速度 Terminal Velocity
      - 优先级 Priority
      - 流体摩擦 Fluid Friction
      - 水体积 Water Volume 决定体积是否包含水
      - 接触时的物理影响
  - 数据分布
  
  

  

