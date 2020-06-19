
## 蓝图 BluePrint
- 蓝图可以让普通Actor动起来，因为它有一个EventGraph可以处理事件。
- 构造过程Contstruction Script
- 如何把球压在地面上
- 如何让BP中定义的Coponent变量在Actor的Default中显示变量
- 给小球滚动时添加声音
- Make Literal xxx
- AddImpose
  - Vel Change的用法
- Tick中deltaseconds的用法
- Print String
- LineTraceByChannel
  - Draw Debug Type设成Persistent可以把这条射线画出来
  - Trace Channel 跟 Actor的Collision相关
- GetActorLocation
- Break Hit Result
  - Impact Normal
- EventActorBeginOverlap
- Cast to 
- Set Physics Linear Velocity
- Timeline
- SetRelativeLocation
- Set Parameter Value on Material xxx
- Set xxx
- Spawn Sound Attached

- Construction Script
- Select
- Move Component To
- 'D' Delay
- 循环
- 动作映射，用什么键控制什么动作， 动作需要先添加一下。Project Settings -> Inputs -> Action Mappings
- PlayerController
- Camera
  - auto active for player

- 如何把别的内容添加到一个BP

- Get Actors of Class
- ForEachLoop
- Swith On xxx
- Add 把元素添加到数组类variable中
- 选中多个，ctrl+w复制一份
- 按住ctrl，把variable拖到bp event graph编辑器中
- Box Collision Component
- DestroyActor
- Bind Event to OnDestroyed
- Delay
- Set Volume Multiplier设置音量
- Set Pitch Multiplier设置音调
- Get Velocity
- Vector Length
- / 除以
- Clamp
- Add Static Mesh Component
- Promote to Variable 存成变量

- Construction Script
  - SetRelativeLocation
  - GetRelativeLocation
  - Find Look at Location

- GameMode 统筹整个游戏的游戏数据
- Player Control Class
- Word Settings -> 设置GameMode override为自定义GameMode

- BluePrint->Enumeration

- Edit -> Extrude -> Subtract

## Modeling

## Material
'T' Texture Sample
  Alpha Channel
Multiply
'L' Lerp
'V' vector
scalar parameter 可以被代码修改

## 声音
- 添加wav文件到content browser
- 右键create cue创建Sound Cue
- Spawn Sound Attached
- Sound Cue编辑界面 -> Output节点 -> Sound -> Volume Multiplier调节声音大小

- 节点分类： 执行节点，读取节点，事件节点
- 节点注释,注释组，注释框
- 情境关联

## Basic Operation
- 'E' 旋转
- 编辑器中可以搜索类，搜索BP操作，搜索选项等。比如在worldSettings界面，可以搜GameMode，界面就只会出现GameMode的设置
- 查找引用

## 快捷键
- alt+shift+r查找引用
- ctrl+B跳到资源
- alt+shift+f查找变量引用
