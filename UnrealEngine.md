# Unreal Engine
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
    - 动画编辑工具
      - 状态机，混合空间，正向和逆向运动学，物理驱动的布娃娃效果动画，同步预览动画
    - 可编制脚本的骨架绑定系统
  - 动画蓝图
  - LiveLink数据流送
  - Take Recorder
  - Sequencer：专业动画
- 渲染、光照和材质
  - 前向渲染
  - 
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




功能
============================================
- nDisplay
显示多多个屏幕上。

- LiveLink
https://blog.csdn.net/lulongfei172006/article/details/80427866
https://docs.unrealengine.com/en-US/Engine/Animation/LiveLinkPlugin/ConnectingUnrealEngine4toMayawithLiveLink/index.html



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
  
  

  

