# Maya5编程全攻略

## Maya架构
```
- 数据流模型 DG， 技术上讲基于一种推拉模型，而不是基于严格的数据流模型。
- 数据及其操作被封装为节点， 一个节点含有任意数目的插槽，其中含有Maya使用的数据，节点也包含一个操作符，用于对其数据进行处理。
- 场景就是DG
- Hypergraph
- 节点类型

  - 常见(注意实际中的命名)
    - time
    - transform
    - mesh
    - tweaks
    - sets
    - groupParts
    - groupId
    - locator ~~loc_anim_jawShape~~
    - joint
    - skinCluster
    - blendShape
    - blendWeighted ~~BW_blendShape1_Eyebrows_Frown_R~~
    - blendColors
    - follicle ~~eyelashes_dn_L_01_flcl~~
    - parentConstraint
    - pointConstraint
    - orientConstraint
    - animConstraint
    - clusterHandle
    - cluster ~~Louise_L_eye1_cls2~~
    - nurbsCurve ~~curveShapexx~~
    - nurbsSurface 
      - 本身作为控制器连接到multDoubleLinear再连到blendshape上。
    - multDoubleLinear ~~mulBS_Jaw_Up~~
      - 以mulBS_Jaw_Up为例，它有两个输入1和输入2
    - multiplyDivide
    - reverse
    - animCurveUL
    - animCurveUU ~~UU_blendShape1_Eyebrows_Frown_R~~
    - animCurveTU
    - unitConversion
    - dagPose ~~bindPose~~
    - lambert
    - dx11Shader
    - file
    - place2dTexture
    - surfaceShader
    - shadingEngine
    - bump2d
    - setRange
    - condition
    - blinn
    - cameraView
    
  不常见
    - sequenceManager
    - hardwareRenderingGlobals
    - renderPartition
    - renderGlobalsList
    - defaultLightList
    - defaultShaderList
    - defaultRenderingList
    - defaultRenderUtilityList
    - defaultTextureList
    - postProcessList
    - lightList
    - materialInfo
    - lightLinker
    - shapeEditorManager
    - displayLayer

- 节点属性
  - 固定属性 input / output
  - 每个节点都内部存储其属性数据
  - 每个节点都有一个计算函数compute
  - 属性是节点中的一个占位符
  - 属性类型
  - 复合属性
    - 子属性，父属性
  - 属性数组
  - 锁定属性
    - setAttr -l true "node.attr";
  - 显示keyable的属性
    - listAttr -keyable "node_name";
- 计算函数
  - 输入和输出是函数？
  - 计算函数永远不会从其他节点或位置获得信息

- 建立属性之间的关系
  - attributeAffects(a, b) // a affects b
  
- 连接节点
- DAG
  - DAG路径
  
- DG更新
  - 推拉式

- 节点编辑器操作
  - 快捷键
  - 基本操作
  - 蓝色箭头，黄色箭头，绿色箭头，粉色箭头，灰色箭头
  - 如何查看一个节点的输入/输出
  
- 一般知识
  - ShapeOrig与Shape的区别， Shape往往是在Orig基础上加了tweak， 加了skinCluster
  - 同样是transform类型，图标会不一样，如何做到的
  - blendshape连到blendshape上
  - blendshape连到skinCluster上
```  
## Maya MEL
### 概述
- 类c语言
### MEL编程语言
- 变量：类型，数组，vector，boolean， 字符串， int, float
- 操作符：关系，逻辑，算数
- 控制:条件，循环
- 脚本：存储脚本，自定义脚本目录 MAYA_SCRIPT_PATH环境变量
- 命令模式:创建，查询，编辑
- 与C的区别：
  - 变量使用时需要加$, 比如定义时需要string $name;
  - 数组： string $arr[]; $arr[size($arr)] = 5;
- 调试:
  - trace 
  - print
- 显示警告和错误
  - warning
  - error
- 确定类型
  - whatIs
- maya.env文件
### 脚本
### 对象
- 基础知识
```
ls
ls -sl
ls -type
delete nurbsCone1;
rename a b;
`objectExists b`;
objectType b;
whatIs // 返回mel语言类型


获得到给定节点的所有可能路径的完整清单
string $paths = `ls -allPaths $nodePath`;

迭代前面的$paths, 然后跟$nodePath去匹配.

获得实例索引。获取目标节点的实例索引，这一点很重要。

string $attr = $toNode + ".worldInverseMatrix["+$instance+"];";
$mtx = `getAttr $attr`;

```
- 层次结构
```
- 创建和导航对象的层次结构
创建新变换
group -n topTransform nurbsSphere1; // 创建transform节点并命名为topTransform，并把nurbsSphere1添加为子节点
parent nurbsCone1 topTransform; // 把nurbsCone1加为topTransform的子节点
move -relative 0 3 0 topTransform； // 所有的子节点都会受到影响。
inheritTransform -off nurbsCone1; // 让nurbsCone1不受topTransform影响
listRelatives topTransform; // 查看topTransform的所有子节点
listRelatives -allDescendents topTransform; // 列出所有子节点
listRelatives -shapes nurbsSphere1; // 列出所有子节点
listRelatives -parent nurbsSphereSphape1; // 列出父节点
reorder -front nurbsCone1; // 更改兄弟的顺序
parent -world nurbsCone1; 或者 ungroup -world nurbsCone1; // 将一个节点从其父节点断开
```
- Transform节点
```
move -relative 3 0 0 nurbsSphere1;
看起来是nurbsSphere1移动了，事实上是nurbsSphereShape1移动了。 nurbsSphere1节点只是含有移动它的子节点的信息。

scale -absolute 1 0.5 1 nurbsSphere1;
rotate -relative 45deg 0 0 nurbsSphere1;
xform -relative -translation xxx;

空间
  局部空间
  世界空间
转换矩阵
worldPoint = point x fingerTransform
arm|hand|finger|finger|fingerShape

finalTransform=fingerTransform x handTransform x armTransform
worldPoint = point x finalTransform;

matrix $mtx[4][4] = `xform -query -matrix nurbsSphere1`; // 报错
float $mat[] = `xform -query -matrix nurbsSphere1`; // 正确。 返回transform节点的当前转换矩阵。行主存的形式
float $mat[] = `xform -query -worldSpace -matrix nurbsSphere1;` // 返回局部到世界空间转换矩阵
```


- 属性
所有场景数据都存储在每个节点的属性中，经常需要访问和编辑属性。
```
getAttr xxx.xxx;
setAttr xxx;

动态属性
$objs = `ls -sl`;
for ($obj in $objs)
  addAttr -longName "points" -attributeType int $obj; // 属性编辑器中的Extra Attributes按钮点开来可找到。

判断属性是否存在
attributeExists("points", $obj);

要使动态属性在Channel Box中显示，必须使属性变得可关键帧化。
setAttr -keyable true nurbsSphere1.points;

删除动态属性
deleteAttr nurbsSphere1.points;

重新命名动态属性
renameAttr nurbsSphere1.points boost;

不能删除或者重命名原先存在的属性。

属性信息
listAttr nurbsSphere1; // 返回string[]
listAttr -userDefined nurbsSphere1; // 列出动态属性
listAttr -keyable nurbsSphere1; // 可关键帧化的属性

获取属性的一般信息
getAttr -type nurbsSphere1.points; // Result : long
getAttr -keyable nurbsSphere1.points; // Result : 0

要获取有关某属性的其他信息
attributeQuery -node nurbsSphere1 -hidden points; // Result : 0
attributeQuery -node nurbsSphere1 -rangeExists points; // Result : 0
```

### 动画
创建、编辑和删除关键帧，通过以程序方式生成关键帧，可以创建复杂的动画。
- 时间
```
时间单位 Working Units， 默认情况下为Film[24fps]
修改时间单位会导致动画缩放。
currentUnit -query -time;
currentUnit -time "min"; // 设置时间为分钟
currentUnit -time "min" -updateAnimation fales; // 关闭自动更改关键点位置。

// 事先不知道工作时间单位使什么。
比如currentTime 10; 会导致产生的效果不符合预期，如果工作时间单位跟自己想的不一样的话。

// 有个办法是设置时间时添加单位
currentTime 2sec;
currentTime 1.25hour;

// 针对时间获取属性。
currentTime 5;
getAttr sphere.translateX;
getAttr -time 10 sphere.translateX;

setAttr仅运训设置当前时间的属性值。

更改时间会导致整个场景进行更新，更新的代价非常大。

可以更改当前时间而不更新场景。
float $cTime = `currentTime -q`;
currentTime -update false 10;
setAttr sphere.translateX 23.4;
currentTime -update false $cTime;

新建場景
file -f new;

获取控制器参数上下限
transformLimits -q -ty cs_rig_n0va_real:RightBrow_inn_raisef_ctrl; // 结果: 0.1 1 //

检查哪些控制器可以被设值
getAttr -settable RightBrow_inn_raisef_ctrl.translateX; // 結果: 1
```
- 播放
```
play;
play -forward false; //反向播放
play -q -state; // 1 表示正在播放
play -state off; // 停止播放
playbackOptions -minTime 12 -maxTime 20; // 修改播放范围，动画范围保持不变
playbackOptions -ast 12 -aet 20; // 修改动画范围
playbackOptions -loop "oscillate";
playbackOptions -playbackSpeed 0.5; // 修改播放速度
playblast -file test.mov; // 播放预览
```
- 动画曲线
```
一条动画曲线由一组控制点及其相关切线组成。

插值方式可定义。

Maya的各种曲线编辑工具。

获取与给定节点相关的所有动画节点。
keyframe -query -name ball;

获取给定属性的动画节点
keyframe -query -name ball.translateX;

给定节点是不是一个动画曲线
isAnimCurve($node)

要确定一个节点是否时可动画化的
listAnimatable -type ball; // Result: transform

确定一个节点的哪些属性是可以动画化的
listAnimatable ball;

获取一个已被动画化的属性的值
keyframe -query -t 250 -eval ball.translateX; // 求值。 -eval标记可以快速尝试不同时间的动画曲线，而不会导致DG在曲线的所有输入连接上进行更新

无穷性值
setInfinity -query -preInfinite ball.translateX;

创建关键点
setKeyframe -t 1 -value -5 ball.translateX;
setKeyframe -t 48 -value 5 ball.translateX;



查询哪个动画曲线现在控制着translateX
keyframe -query -name ball.translateX;

驱动关键帧： 想要用translateX来控制ball的缩放时，需要使用驱动关键帧
setDrivenKeyframe -driverValue 0 -value 0 -currentDriver ball.translateX ball.scaleX;
setDrivenKeyframe -driverValue 5 -value 1 ball.scaleX;

删除动画曲线
string $nodes[]=`keyframe -q -name ball.translateX`;
delete $nodes[0];

插入关键点
setKeyframe -insert -time 24 ball.translateX;

查询和编辑动画曲线中的关键点

关键帧数
keyframe -q -keyframeCount ball.translateX;

选择的关键帧数
keyframe -q -selected -keyframeCount ball.translateX;

获取所选关键点的范围
keyframe -q -selected -timeChange ball.translateX; // Result: 48

获取所选关键的实际值
keyframe -q -selected -valueChange ball.translateX;

编辑关键点
keyframe -edit -t 48 -timeChange 20 ball.translateX; // 将关键点从48移动到20.

相对变化
keyframe -e -t 20 -relative -timeChange 5 ball.translateX; // shi

： 设置索引区间
keyframe -e -index "1:20" -vc 2.5 ball.translateX;

缩放时间
scaleKey -time ":" -valueScale 0.5 ball.translateX;
scaleKey -timeScale 0.5 ball.translateX;
scaleKey -timePivot xx -timeScale 2 ball.translateX; // 轴心

使关键点落到帧号上
snapKey -timeMultiple 1 ball.translateX;

中间关键帧




入切线和出切线
切线性质： 类型(step, flat, )，角度，加权和锁定

开始停着，在动画末尾猛然跳到终点。
keyTangent -index 0 -outTangentType step ball.translateX; 

缓缓启动，然后加速移向终点。
keyTangent -index 0 -outTangentType flat ball.translateX;

将切线恢复原始状态
keyTangent -index 0 -outTangentType spline ball.translateX;

要改回原来的状态可能需要同时更改两条切线。
keyTangent -index 0 -inTangentType spline ball.translateX;

现在将一条切线旋转45度  inAngle
keyTangent -index 0 -inAngle 45 ball.translateX;

相对旋转
keyTangent -index 0 -relative -inAngle 15 ball.translateX;

查询出切线作为一个单位矢量的方向
keyTangent -index 0 -q -ox -oy ball.translateX; // 0.15, 0.98
keyTangent -index 0 -q -ix -iy ball.translateX; // 0.15, 0.98

解除对入切线和出切线的锁定，以便它们能够独立旋转
keyTangent -index 0 -lock false ball.translateX;

为了给切线加权（修改切线长度）, 必须将整个动画曲线转化为使用加权切线。
keyTangent -e -weightedTangents yes ball.translateX;

撤销加权的切线会失去它的加权信息，曲线会变形。

给第二个关键点切线加权
keyTangent -index 1 -inWeight 20 ball.translateX;
keyTangent -index 1 -lock false -weightLock no ball.translateX;

Tangents设置
keyTangent -q -global -inTangentType;
keyTangent -q -global -outTangentType;
keyTangent -q -global -weightedTangents; // Query automatic weighting; 是否使用加权切线
keyTangent -global -inTangentType flat;
keyTangent -global -weightedTangents yes;

关键点剪贴版

// 将第一个关键点复制插入到关键帧12.
copyKey -index 0 ball.translateX;
pasteKey -t 12 -option insert ball.translateX;

cutKey -index 1 -option keys ball.translateX;
cutKey -index 1 -clear ball.translateX;

page 127 获取动画曲线的所有信息 PrintAnim

```
- 骨架
  - 蒙皮skinning或enveloping
```

所有定位都在世界空间中给出。
joint; // 在原点生成一个关节joint1。
joint -position 0 0 10; // 创建另一个关节joint2, 并使得它成为joint1的子关节。
insertJoint joint1; // 在joint2和joint1之间插入一个新的关节joint3。但它不允许指定初始节点的详细位置。
joint -edit -position 0 0 5 joint3; // 使用joint -e -position编辑位置。 但这个命令会让joint3的子节点joint2也向下移动5.
undo;
joint -e -position 0 0 5 -component joint3; // 使用-component可以避免子关节的移动。

// 或许相对于其关节来指定一个关节的位置更为方便，使用-relative标记
joint -e -relative -position 5 0 0 joint2.

// 旋转自由度，默认有3个。

joint -query -degreeOfFreedom joint1; // 查询旋转自由度 xyz

joint -e -degreeOfFreedom "y" joint1; // 将旋转限制在y轴

joint -e -limitSwitchY yes joint1; // 激活y轴的旋转限制

joint -q -limitY joint1; // 查询当前的限制是什么

joint -e -limitY 0deg 90deg joint1; // 限制在0-90度的旋转

rotate 0 200deg 0 joint1; // 该关节最多旋转90度，虽然指定了200deg.

// 所有使用joint命令的关节旋转都是相对的。 只有关节的定位才可以在绝对坐标中进行。

joint -e -angleY 10deg joint3; // 相对于其父关节来旋转一个关节

// 重要： 查询一个节点的当前旋转
xform -q -rotation joint3;
xform -q -translation joint3;
xform -q -relative -scale joint3;

// 删除一个节点
removeJoint joint3;

// 删除一个节点及其子节点
delete joint3;

// 重要： p135 OutputJoints函数

// 重要： p140 ScaleSkeleton脚本

float $pos[] = `joint -q -relative -position $child`; // 获取关节节点相对于其父节点的当前位置
$pos[0] *= $scale;
$pos[1] *= $scale;
$pos[2] *= $scale;
joint -e -relative -position $pos[0] $pos[1] $pos[2] $child;

// 重要： p142 CopySkeletonMotion脚本
xform -query -worldSpace -translation $srcRootNode;

listRelatives -fullPath -children -allDescendents $srcRootNode;

string $inPlugs[] = `listConnections -plugs yes -destination yes joint1.translateX`;

if(size($inPlugs)){
string $tokens[];
tokenize $inPlugs[0] "." $tokens;
string $inNode = $tokens[0];
string $inAttr = $tokens[1];
string $dupInNodes[] = `duplicate -upstreamNodes $inNode`;
connectAttr -force ($dupInNodes[0]+"."+"translateX") ($destNode+"."+"translateX");
}else{
getAttr xxx;
setAttr xxx;
}

move -worldSpace x y z $node;
```

- 运动路径
```
- string $mpNode = `pathAnimation -curve myCurve muCone`;

- addDoubleLinear节点

- 得到曲线最大的u参数值
$maxUValue = `getAttr muCurveShape.minMaxValue.maxValue`;

$endTime = `playbackOptions -q -animationEndTime`;

setKeyframe -t $endTime -value $maxUValue -attribute uValue $mpNode;

//创建路径的开始和结束时间
pathAnimation -startTimeU 0 -endTimeU 48 -curve myCurve myCone;

pathAnimation -e -follow yes myCone; // 锥尖的方向指向路径的方向

pathAnimation -e -bank yes myCone; // 倾斜转弯

pathAnimation -e -followAnis x myCone; // 进出转弯

pathAnimation -e -bankScale 0.5 myCone; // 减少倾斜

flow myCone; // 使锥体改变形体以便沿路径扭动。

```


### 图形用户界面
- 概述
```
创建空窗口
window -title "test";
paneLayout -configuration "horizontal2";
columnLayout；
  floatSliderGrp -label "TranslateX" -min -100 -max 100;
  floatFieldGrp -label "Scale" -numberOfFields 3;
  setParent ..;
frameLayout -labelVisible false;
  outlinerPanel; // Maya自己的大纲视图
showWindow;
```
- 基本概念
```
window; // 创建窗口。 默认状态下使不可见的。
window -e -visiable true window1; // 使窗口可见
window -e -visible yes window1;
showWindow window1; // 使窗口可见
window -e -widthHeight 200 100 window1;
window -q -numberOfMenus window1; // 查询菜单数
button -label "First" myButton; // 创建按钮

// 布局
columnLayout
frameLayout
button -label "Second" -parent myWindow|myLayout;

// 层次
// 界面元素组成层次结构， 窗口始终位于层次结构的顶端。只要完整路径不同，两个元素可以用相同的名称。
// 当创建一个界面元素时，会默认成为自己类型元素的默认父元素。
// setParent命令

// 执行命令
window;
columnLayout;
button -label "Click" -command "sphere";               // button
textField -text "xxx" -enterCommand "myText(\"#1\")"； // textField #1是元素当前值的速记符号
showWindow;
```
- 窗口
```
// 判断窗口是否存在
window -exists myWindow; 

// 删除窗口
deleteUI myWindow; 

// 窗口自适应
window -title "xxx" -resizeToFitChildren true myWindow; // 根据Child columnLayout等调整自己的大小。
columnLayout -adjustableColumn true;

// 设置固定尺寸
window -width 999 -height 000 -topEdge 999 -leftEdge 999 -topLeftCorner 999 999 xxx;

// 清除Maya所存储的窗口的大小和位置信息
windowPref -remove myWindow;

// 使UI调用自定义MEL 函数。
global proc rm_test(){
}
button -label "xxx" -command "rm_test();";

// 模式对话框
promptDialog -message "xxx" -button "OK" -button "Cancel" -defaultButton "OK" -cancelButton "Cancel" -dismissString "Cancel";

// 确认对话框
confirmDialog -message "Do you wnat to delete the object" -button "Yes" -button "No".
```
- 布局

```
// 菜单布局
menuLayout

// 通用命令
layout -edit -width 200 $layoutName;

// ColumnLayout
// RowLayout
// GridLayout
// FormLayout 最灵活的布局
// FrameLayout 折叠和展开的界面
// TabLayout
tabLayout的子元素必须是布局。
// ScrollLayout
// MenuBarLayout
setParnet -topLevel;
menuBarLayout;
```
- 控件
```
control -e -width 23 $controlName;
control -q -parent $controlName;
control -exists $controlName;

菜单
menu, menuItem

按钮
button

创建一个显示图像或图标的按钮
symbolButton -image "sphere.xpm" -command "sphere;";

显示标记和图像
iconTextButton

复选框
checkbox
symbolCheckBox
iconTextCheckBox

单选按钮
radioCollection
iconTextRadioButton
iconTextRadioCollection

文本
text静态文本
textField
scrollField

清单
textScrollList -allowMultipleSelection  page175

组
floatSliderGrp 
colorIndexSliderGrp
colorSliderGrp
floatFieldGrp
floatSliderGrp
intFieldGrp
intSliderGrp
radioButtonGrp
textFieldGrp

图像
picture显示.xpm和.bmp图像。
image命令不仅可以显示这些格式，还可以显示更多格式。 这个测试可以使用全路径的图片。

面板
paneLayout -configuration "vertical2";
有个想法是可以把imagePlane放在某个较远的地方让后把modelPanel作为一个panel放在GUI里

工具
激活一个工具而不是执行一条MEL命令。
toolCollection
toolButton
```

- 连接控制
图形用户界面元素或组反映一个节点属性的当前值。

```
属性组
attrColorSliderGrp 获取节点属性并同步。

如何获取当前帧呢？
attrFieldSliderGrp -attribute "timer1.outTime" -min 0 -max 44594 -label xxx;

NameField

其他元素
connectControl命令
string $grp = `floatFieldGrp -label "Scale:" -numberOfFields 3`;
connectControl -index 2 $grp sphere1.scaleX;
connectControl -index 3 $grp sphere1.scaleX;
connectControl -index 4 $grp sphere1.scaleX;
// 注意1对应于"Scale:"

用户反馈
  帮助
  button -annotation ""

  显示进程
  waitCursor -state on;
  waitCursor -state off;
  进度条 gressWindow -title "Working" -progress $amount -status "Complete: 0%" -Interruptable true; // page 186
  
  pause -seconds 1;
```

### 表达式

过程式动画化。通过MEL命令来定义属性的值。通过程序来而不是通过关键帧来控制动画。表达式可以创建复杂的动画而不涉及或很少涉及手动操作。这是一种将属性动画化的强大方法，这样就可以完全自由地使用所需的任何MEL命令或者语句。这里的表达式是指使用MEL语句来将属性动画化。
- 基本
```
有一些效果可以交给表达式去产生
摆动
闪烁
磁体


```
- 表达式使用注意事项
```
仅对属性赋值
支持的属性类型 float, int核和boolean。
避免使用setAttr和getAttr命令
所连接的属性
sphere.scaleX = 23; // 如果scaleX已经动画化，也就是有连接，那么会报错。
需要用disconnectAttr中断至scaleX属性的连接。

表达式是DG节点。它连接表达式所控制的各个属性。 expression节点实际存储着你的表达式。

自动删除
如何避免自动删除。 p198

```
- 调试表达式
- 粒子表达式
- 高级表达式
```
关键帧动画与表达式一起使用
expression -string "plane.translateY = plane.animTranslateY" -name "Turb"; / 创建了名为Turb的表达式
```

### Maya C++ API (page 208)
- 介绍
```
此API可修改的领域和功能：
命令
DG节点： 自定义节点， 或者 扩展专用的DG节点
工具/语境
文件转换器，变形器，着色器，操纵器，定位器，动力场，发射器，形体，解算器
```
- 类
```
抽象层: 开发人员 C++API Maya核心

类： 层次结构
M, Maya类 MObject, MPoint, M3dView
MPx 代理对象 MPxNode
MIt 迭代类 MItDag, MItMeshEdge
MFn 函数集 MFnMesh, MFnDagNode

传统方法： 面向对象方法都包含数据和作用于数据的函数。

Maya方法与普通面向对象方法的区别： 数据与函数分离。
数据层次结构： 用单独的类来存储数据。数据层次结构。
函数集层次结构： 对应于数据层次结构。

CarObj carData;
VehicleFn speedyFn;
speedyFn.setObject(&carData);
speedyFn.drive();

所有数据均是通过类MObject访问的。只有MObject类知道这个层次结构。实际指向Maya核心内一个句柄。UE感觉也是这么实现的。
必须使用函数集层次结构来处理Maya的隐藏数据对象。
```
- MObject
```
MObject仅仅是指向核心内另一个对象的一个句柄。
Maya拥有所有的数据，而你不拥有它内部的任何数据。
```

- MFn函数集
```
MFn函数集用来创建、编辑和删除数据。 比如要创建transform节点，可以这样。
MFnTransform transformFn;
MObject transformObj = transformFn.create();
MFnBase -> MFnDependencyNode -> MFnDagNode -> MFnTransform

MFnDagNode dagFn(transformObj);
dagFn.name();

MFnPointLight pointLightFn;
MObject pointObj = pointLightFn.create();
MFnNurbsSurface surfaceFn(pointObj); // 需要添加错误检查和报告
double area = surfaceFn.area();

自定义函数集
新的类只能调用其即类已经实现的函数，不能提供自己的函数，因为你不能处理任何实际的Maya数据。从这些函数集中进行派生就没有任何意义。
```

- 开发插件
```
为了给Maya添加自定义节点、命令等，需要创建插件。

插件向导。

写插件。
MStatus initializePlugin(MObject obj){
  MFnPlugin pluginFn(obj, "developer", "version", "Any");
  pluginFn.registerCommand("goRolling", GoRollingCmd::creator, GoRollingCmd::newSyntax);
  pluginFn.registerNode(xxx);
}
MStatus uninitializePlugin(Mobject obj){
  pluginFn.deregisterCommand("goRolling");
  pluginFn.deregisterNode(RollingNode::id);
}
如何调试插件??

错误，报告
MStatus， 

插件相关命令
加载/卸载pluginInfo -q xxx;
unloadPlugin full_file_path;
loadPlugin full_file_path;
MAYA_PLUG_IN_PATH
```

- 命令
```
注册命令


简单命令
doIt和creator两个函数需要实现。


第一个例子。命令无参数
MDagPath dagPath; // MDagPath是MObject的一种，也就是也是一种对内部
MFnNurbsCurve curveFn;
MItSelectionList iter(selection, MFn::kNurbsCurve);
for(;!it.er.isDone(); iter.next())
{
  iter.getDagPath(dagPath);
  curveFn.setObject(dagPath); // dagPath is a MObject
  
  double tStart, tEnd;
  curveFn.getKnotDomain(tStart, tEnd);
  ...
  curveFn.getPointAtParam(t, pt, MSpace::kWorld); // 获取位于世界空间中的位置。
  MGlobal::executeCommand();
};


第二个例子。 带参数命令， -number 10 -radius 1
int nPosts = 5;
unsigned index;
index  = args.flagIndex("n", "number");
if(MArgList::kInvalidArgIndex != index)
  args.get(index+1, nPosts);

第三个例子。 带很多参数命令时，参数的另外的写法, MSyntax。
MSyntax 指定命令的语法对象
1.Command中添加静态函数static MSyntax newSyntax()
2.插件注册时需要提供这个静态函数
stat = pluginFn.registerCommand("posts3", Posts3Cmd::creator, Posts3Cmd::newSyntax);
3.doIt函数中用MArgDatabase及MSyntax对象来parse参数
MArgDatabase argData(syntax(), args);
if(argData.isFlagSet(numberFlag))
  argData.getFlagArgument(numberFlag, 0, nPosts);

第四个例子。提供帮助 help posts3。先pass。
1.自动帮助。 前面加了Msyntax后，就能调用help。
2.创建和显示自己的帮助。 -h/-help。
if(argData.isFlagSet(helpFlag)){
  setResult(helpText);
  return MS::kSuccess;
}

撤销/重做机制。
1.撤销队列。只能保存一条可能的历史路径。撤销队列与场景一起保存。
2.MPxCommand支持
virtual MStatus undoIt();
virtual MStatus redoIt();
virtual bool isUndoable()const; //默认false， 如果是false, 命令执行完马上删除。如果是true, 
2.1. 用creator函数创建命令实例
2.2. 调用命令对象的doIt函数
2.3. 调用isUndoable函数判断是否把此实例放入撤销队列中。
**如果命令能以某种方式改变Maya状态， 它就必须提供撤销和重做。**
**不支持撤销的命令称为action, 比如计数命令， 执行时不会创建，修改或更改Maya状态。**
3.新数据成员MDGModifier dgMod; // 这个类用于创建、删除和更改依赖图上的节点。虽然MFn系列也能做到这些，但是MDGModifier类最大的好处就是它能自动地为所有操作提供撤销和重做。
// 伪代码
doIt(...){
  dgMod.commandToExecute("cylinder"); // 实际没有执行。
  return redoIt(); / 实际执行。
}
redoIt(...){
  return dgMod.doIt();
}
undoIt(...){
  return dgMod.undoIt();
}
4.redoIt函数和doIt是一样的。


编辑和查询 -e -q。
//伪代码
::newSyntax(){
  MSyntax syntax;
  syntax.addFlag(xx, xx, MSyntax::kLong);
  ...
  syntax.enableQuery(true);
  syntax.enableEdit(true);
  return syntax;
}
例子p255 : 如何实现undo, redo

void xxxCmd::getTime(){
  MFnTransform transformFn;
  transformFn.setObject(hourHandPath);
  
  MEulerRotation rot;
  transformFn.getRotation(rot);
}

void xxxCmd::setTime(){
  transformFn.setRotation(MEulerRotation(MVector(0, xx, 0)));
}

```
- 节点
  - 一、GoRolling插件。开发自定义节点，使它能够根据轮子的位置来自动旋转轮子。创建一个命令和一个节点。
  ```

  {
      1. goRolling命令用于创建节点RollingNode, 将其插入DG， 并进行所有必要的属性连接，节点控制着对象的实际旋转。
      {
          //伪码
          doIt(){
            ...
            iter.getDagPath(dagPath);
            transformFn.setObject(dagPath);
            // 对于selection中每个transform，都会创建一个新的RollingNode
            MObject rollNodeObj = dgMod.createNode("RollingNode");
            MFnDependencyNode depNodeFn(rollNodeObj);
            // 通过节点的接头（plug）来访问节点的属性值。
            // MDGModifier可以把两个plug连接起来。
            dgMod.connect(transformFn.findPlug("translateX"), depNodeFn.findPlug("distance"));
            dgMod.connect(depNodeFn.findPlug("rotation"), transformFn.findPlug("rotateZ"));
            return redoIt();
          }
          redoIt(){
            return dgMod.doIt();
          }
          undoIt(){
            return dgMod.undoIt();
          }
      }


      2. RollingNode节点
      {
          class RollingNode : public MPxNode
          {
            virtual MStatus compute(const MPlug &plug, MDataBlock &data);
            static void *creator();
            static MStatus initialize();

            static MObject distance;
            static MObject radius;
            static MObject rotation;

            static MTypeId id; // 每种类型的节点都有一个惟一的标识符，这样Maya就能知道如何创建这个节点以及从磁盘上如何保存和恢复节点数据。这个id确定了以后最好不要变，否则可能会导致打不开老的场景
          }

          **compute函数带有两个参数：第一个参数是MPlug类型，它指示哪个接头plug（节点）要被重新计算，第二个参数是MDataBlock类型，它包含了节点当前要使用的所有数据。**
          **一个节点可以有很多个输出属性。由于仅计算节点的输出属性，而它们可能并不都需要更新，所以Maya会对节点的每个输出属性逐一调用compute函数。因此开发人员必须检查时节点的哪个输出属性被请求更新。**
          **plug是MObject的继承类，可以直接跟自定义属性判断是否相等 plug == RollingNode::rotation**
          // 伪代码
          compute(plug, data){
            if(plug == rotation){

            }
          }

          **当重新计算rotation属性的请求时， 首先获取节点的当前输入属性值。MDataBlock.inputValue函数读取数据。MDataBlock.outputValue函数只允许开发人员写入而不能够读取它。尽管Maya没有对节点的输入属性和输出属性进行区分，但是通过使用独立的输入和输出函数可以更有效地提取和维护节点数据**
          // 伪码
          compute(plug, data)
          {
            // 获取input属性值distance和radius
            MDataHandle distData = data.inputValue(distance);
            MDataHandle radData = data.inputValuye(radius);
            // distance即plug
            // plug + data -> data handle -> value
            double dist = distData.asDouble();
            double rad = radData.asDouble();

            // 将计算结果存储到rotation属性中，需要一个句柄对象指向这个属性。
            MDataHandle rotData = data.outputValue(rotation); // 返回一个可以改写节点属性值的MDataHandle对象。
            rotData.set(-dist/rad);

            // 接下去时将节点的接头设置为"干净"标记，这就告诉Maya此接头已经被重新计算过了，现在所含的是的正确值。
            data.setClean(plug);
          }

          creator函数
          void *RollingNode::creator(){
            return new RollingNode();
          }

          **initialize函数：用于设定节点的所有属性。由于节点的distance, radius, rotation属性都是静态成员，所有的节点实例对象共享它们的一个拷贝。这是因为属性并没有真正保存每个节点属性的数据。
          相反，它们只是作为创建每个节点将要使用的属性数据的蓝图。从技术上讲，一个节点的属性数据存储在一个MDataBlock中，并通过使用MPlug来获取和设置属性的值。在C++ API的语境中，一个属性是通过使用MAttribute类及其派生类来创建和编辑的。Maya支持的属性值类型很广泛，既包括简单的数值类型（如bool, float, int）等，还包括更为复杂的复合类型数据。**
          RollingNode::initialize(){
            // MFnxxxAttribute： 定义属性。通过使用MFn功能的create函数来生成。
            MFnNumetricAttribute nAttr;
            distance = nAttr.create("distance", "dist", MFnNumetricData::kDouble, 0.0);
            radius = nAttr.create("radious", "rad", MFnNumetricData::kDouble, 0.0);
            MFnUnitAttribute uAttr;
            rotation = uAttr.create("rotation", "rot", MFnUnitAttribute::kAngle, 0.0);

            // 创建了所有属性后，通过调用MPxNode类的addAttribute函数将其添加到节点中。
            addAttribute(distance);
            addAttribute(radius);
            addAttribute(rotation);

            // 为了让Maya知道这些属性是如何相互影响的，必须使用attributeAffects函数来显式地声明这一点。
            attributeAffects(distance, rotation);
            attributeAffects(radius, rotation);
          }

          // 注册命令的地方添加注册节点的代码
          pluginFn.registerNode("RollingNode", RollingNode::id, RollingNode::creator, RollingNode::initialize);
          pluginFn.deregisterNode(RollingNode::id);
      }
  }
  ```
  - 二、Melt插件, 模拟物体融化的效果。演示复杂属性。演示构造历史。
  ```
  有句话需要理解一下： nurbsSphereShape1节点存储了最终的NURBS曲面。这个shape节点还要负责显示其形状。 ？？plug or MObject属性只是数据的描述， 具体数据存放在MDataBlock中。？
  
  复杂属性： 属性存储一个完整的NURBS曲面。这个NURBS曲面属性能够被存储在节点中并且可以连接到其他节点。所以NURBS曲面信息可以通过连接从一个节点有效地传递到到另一个节点中。
  
  构造历史：
  命令会自动地动画化节点的属性。
  
  // 命令代码
  doIt(){
    MSelectionList selection;
    MGlobal::getActiveSelectionList(selection);
    //获取动画范围
    MTime startTime = MAnimControl::minTime();
    MTime endTime = MAnimControl::maxTime();
    
    / 在选定的对象中迭代所有NURBS对象
    MItSelectionList iter(selection, MFn::kNurbsSurface);
    
    // 在所有NURBS对象中获取形体节点
    MObject shapeNode;
    iter.getDependNode(shapeNode);
    
    // 通过MFnDependencyNode函数集合获取至shape节点的create属性的接头。这个属性中存储了一个NURBS曲面。
    MFnDependencyNode shapeFn(shapeNode);
    MPlug createPlug = shapeFn.findPlug("create"); // 查找Plug
    
    // 输入属性只能由一个输入连接
    MPlugArray srcPlugs;
    createPlug.connectedTo(srcPlugs, true, false); // 获取连接到当前接头的所有接头。连接是没有方向的。
    
    MPlug srcPlug = srcPlugs[0];
    
    // 插入一个melt节点
    MObject meltNode = dgMod.createNode(MeltNode::id);
    
    
    MFnDependencyNode meltFn(meltNode);
    MPlug outputSurfacePlug = meltFn.findPlug("outputSurface");
    MPlug inputSurfacePlug = meltFn.findPlug("inputSurface");
    
    // 断开已有连接
    dgMod.disconnect(srcPlug, createPlug);
    
    // 连接不同节点属性的接头
    dgMod.connect(srcPlug, inputSurfacePlug);
    dgMod.connect(outputSurfacePlug, createPlug);
    
    // 重命名melt节点
    static i = 0;
    MString name = MString("melting")+i++;
    dgMod.renameNode(meltNode, name);
    
    // 为melt节点的amount属性设置一个关键帧。
    // 设置关键帧的方法：
    //  1.手动创建一个animCurve节点并把它连接到amount属性上。然后对animCurve节点使用MFnAnimCurve函数集来设置关键帧。
    //  2.更简单的方法：使用MEL命令setKeyframe来为属性设置关键帧。事先写好MEL语句，然后传给MDGModifier的commandToExecute函数。
    MString cmd;
    cmd = MString("setKeyframe -at amount -t ")+startTime.value() +" -v " + 0.0 + " " + name;
    dgMod.commandToExecute(cmd);
    cmd = MString("setKeyframe -at amount -t ")+endme.value() +" -v " + 1.0 + " " + name;
    dgMod.commandToExecute(cmd);
    
    return dgMod.redoIt();
  }
  redoIt(){
    return dgMod.doIt(); 
  }
  undoIt(){
    return dgMod.undoIt();
  }
  
  // 节点代码
  {
	  MObject MeltNode::inputSurface;
	  MObject MeltNode::outputSurface;
	  MObject MeltNode::amount;

	  // 注意MDataBlock的接口
	  // MDataHandle      inputValue ( const MPlug & plug, MStatus * ReturnStatus = NULL );
	  // MDataHandle      inputValue ( const MObject & attribute, MStatus * ReturnStatus = NULL );

	  MStatus MeltNode::compute(const MPlug &plug, MDataBlock &data){
	    if(plug == outputSurface){
	      //获得属性句柄. plug + data + attribute -> dataHandle
	      MDataHandle amountHnd = data.inputValue(amount);
	      MDataHandle inputSurfaceHnd = data.inputValue(inputSurface);
	      MDataHandle outputSurfaceHnd = data.inputValue(outputSurface);

	      // dataHandle -> value
	      double amt = amountHnd.asDouble();
	      MObject inputSurfaceObj = inputSurfaceHnd.asNurbsSurface();

	      // Maya将一些较为复杂的数据类型（例如NURBS曲面）存储为特殊的集合体数据。为了能够创建和存储这种类型的数据，就需要使用适当的MFnGeometryData函数。

	      // 用MFnNurbsSurfaceData创建新的数据块。
	      MFnNurbsSurfaceData surfaceDataFn;
	      MObject newSurfaceData = surfaceDataFn.create();

	      // 复制最初的输入曲面信息复制到这个新的输出曲面数据块中。
	      MFnNurbsSurface surfaceFn;
	      surfaceFn.copy(inputsurfaceObj, newSurfaceData);

	      surfaceFn.setObject(newSurfaceData); // 为啥一个MFnNurbsSurface的功能集可以设置到一个NurbsSurfaceData的MObject上。
	      MPointArray pts;
	      surfaceFn.getCVs(pts);

	      // 算法更新pts

	      // 将新的控制点更新到NURBS曲面。
	      surfaceFn.setCVs(pts);

	      // 调用MFnNurbsSurface对象的updateSurface函数
	      surfaceFn.updateSurface();
	      把输出NURBS曲面属性设置为新的NURBS曲面数据。
	      outputSurfaceHnd.set(newSurfaceData);

	      data.setClean(plug);   
	    }else
	      stat = MS::kUnknownParameter;
	    return stat;
	  }
	  
	  在MeltNode::initialize()中设置属性可动画化
	  {
	  	MFnNumetricAttribute nAttr;
	  	nAttr.setKeyable(true);
	  	MFnTypeAttribute tAttr;
		tAttr.create("inputSurface", "is", MFnNurbsSurfaceData::kNurbsSurface);
		tAttr.setHidden(true);
		
		// 由于outputSurface始终可以从输入inputSurface计算出来，所以不需要存储在Maya场景中。
		tAttr.setStorable(false);
		
	 }
  }
  
  ```
  - 三、groundShadow插件, MDagModifier类。
  ```
  doIt(){
	  MFn::kGeometric
	  MDagPath geomTransformTransform(geomShapePath);
	  geomTransformTransform.pop(); // 得到transform对象。
	  MFnDagNode geomShapeFn(geomShapePath);
	  MObject newGeomTransformObj = geomShapeFn.duplicate(false, false); // 不要创建实例

	  //MFnDagNode也可以获得层级.
	  MFnDagNode newGeomShapeFn(newGeomTransformObj);
	  newGeomShapeFn.setObject(newGeomShapeFn.child(0)); // 通过MFnDagNode获得子对象

	  //将新创建的形体节点放置到原始形体的父transform节点下面。
	  //MFn --> MObject
	  //DagPath --> MObject 通过geomTransformPath.node();
	  dagMod.reparentNode(newGeomShapeFn.object(), geomTransformPath.node()); // MFn对象获取它所指向的MObject， 及使用dagMod reparent。

	  // 将shape节点添加到着色组。
	  shadingGroupFn.addMember(newGeomShapeFn.object());

	  // 创建groundShadow节点
	  MObject shadowNode = dagMod.MDGModifier::createNode(GroundShadowNode::id);

	  // 为groundShadow节点的各个属性生成接头。
	  MPlug castingSurfacePlug = shadowNodeFn.findPlug("castingSurface");
	  MPlug shadowSurfacePlug = shadowNodeFn.findPlug("shadowSurface");
	  MPlug lightPositionPlug = shadowNodeFn.findPlug("lightPosition");

	  // 判断shape节点类型
	  // geomShapePath.apiType(), MFn::kMesh, MFn::kNurbsSurface
	  MString outGeomPlugName, inGeomPlugName;
	  switch(geomShapePath.apiType()){
		case MFn::kMesh:
		outGeomPlugName = "worldMesh";
		inGeomPlugName = "inMesh";
		break;
		case MFn::kNurbsSurface:
		outGeomPlugName = "worldSpace";
		inGeomPlugName = "create";
		break;
	  };

	  // 几何体输出属性是一个输出接头数组而不是单个输出接头。 为什么会有多个呢？ 每个实例对象都存在一个不同的输出。

	  // 判断当前选择的对象对应着几何体的哪个实例
	  unsigned int instanceNum = geomShapePath.instanceNumber();
	  MPlug's selectAncestorLogicalIndex。
  }
  
  // 节点代码
  compute(plug, data){
  ...
  //对每个点进行更新操作
  MPoint surfPoint;
  double denom, t;
  MItGeometry iter( shadowSurfaceHnd, false);
  for(;!iter.isDone(); iter.next()){
  	surfPoint = iter.position(MSpace::kWorld);
	iter.setPosition(surfPoint, MSpace::kWorld);
  };
  data.setClean(plug);
  }
  ::initialize(){
  	MFnUnitAttribute uAttr; // 距离单位
	groundHeight = uAttr.create("groundHeight", "grnd", MFnUnitAttribute::kDistance, 0.0);
	uAttr.setKeyable(true);
	// 可接受多种不同的数据类型的属性
	MFnGenericAttribute gAttr;
	castingSurface = gAttr.create("castingSurface", "csrf");
	gAttr.addAccept(MFnData::kMesh);
	gAttr.addAccept(MFnData::kNurbsSurface);
	gAttr.setHidden(true);
  }
  
  ```
  - 四、属性
    - 所有的节点属性都是用MFnAttribute类的派生类定义的。
    - Maya利用属性重所定义的信息，来实际创建存在于各个节点内的数据。
    - 一个属性只是一个蓝图，所以从逻辑上推理它只能被定义一次。
    - 复合属性？
    - 数组属性？
  ```
  创建属性
  MFnAttribute::create()
  MPxNode::addAttribute()
 
  复合属性
  用来将其他属性分成组，被分成组的属性称为复合属性的子。而复合属性则称为被分成组的属性的父。
  MFnCompoundAttribute compFn;
  playerAttr = compFn.create("player", "ply");
  playerAttr.addChild(nameAttr);
  playerAttr.addChild(homeRunsAttr);
  addAttribute(nameAttr);
  addAttribute(homeRunsAttr);
  addAttribute(playerAttr);
 
  默认值
  每个属性都有一个默认值。 
  如果在过程中属性始终是默认值，那么不会保存到硬盘上。
  默认值发生变化容易引起问题。
 
  MFnEnumAttribute enumFn;
  MFnNumetricAttribute numFn;
  attr = enumFn.create("days", "d", 0); // create传入默认值
  or
  // numeric fn to create boolean
  attr = numFn.create("active", "act", MFnNumericData::kBoolean); 
  numFn.setDefault(false); // 使用MFnxxx::setDefault
 
  4.属性的性质
  Readable 可以作为连接的源属性 默认true
  Writable 可以作为连接的源属性 默认true
  Connectable                 true
  Storable                    true
  Keyable                     false
  Hidden                      false
  UsedAsColor                 false
  *Cached                      false* 避免调用compute获取值
  *Array                       true*  
  通过将属性设定为数组，就可以使其保存一系列的数据实例。数组属性也称为多值属性。一个属性在默认状态下不是数组
  IndexMatter  索引不应更改    false
  ArrayDataBuilder 使用数组数据构造程序来设定值 false
  Indeterminant    确定是否可以用于求值        false
  DisconnectBehavior 断开连接后的行为          kNothing
  Internal           内部的                   false
  RenderSource       改写渲染取样信息	      false
  
 
  动态属性 p298
  可以添加到节点实例或者所有同类型的节点实例。
  ```
  - 五、compute函数
  ```
  通过一个MDataBlock类型的数据快来获取和设置节点的属性值。只能通过MDataBlock数据快来获取节点所有的属性值，然后再通过所获取的输入属性值计算出输出属性值。
  在compute函数的计算中，不能使用MDataBlock以外的任何其他数据。
  
  在compute函数中不要调用MEL的setAttr命令和getAttr命令。如果在此求值过程中直接或间接请求同一接头的值，都会导致系统进入一个死循坏。防止任何的间接DG重新求值。
  ```
  - 六、接头
  ```
  属性仅仅是在节点内创建数据的一个蓝图。它除了提供如何创建这个数据的规范外不保存任何数据。
  访问数据需要通过接头来操作。接头提供了访问给定节点内的实际数据的一种机制。通过指定一个节点及其属性，就可以创建相应的接头。节点用MFnDependencyNode表示，属性用字符串名字表示。
  MFnDependencyNode nodeFn(ballObj);
  MPlug transxPlg = nodeFn.findPlug("translateX");

  //取值
  double tx;
  transxPlg.getValue(tx);

  //设值
  transxPlg.setValue(double(2.3));

  //默认状态下，取的是当前时间的值。
  //如何获取其他时刻的值 这时要使用MDGContext类。
  MTime t(1.5, MTime::kSeconds);
  MDGContext ctx(t);
  transxPlg.getVlaue(tx, ctx);

  // 改变场景的当前时间，然后再设置属性的值。
  MTime t(1.5,MTime::kSeconds);
  MGlobal::viewFrame(t); // 这个函数常常要求DG进行更新。所以应尽量少用这个函数。

  要为属性创建动画化的值，可以通过调用MEL中的keyframe命令。

  复合接头
  {
	  //用MPlug定位复合属性。
	  MFnDependencyNode nodeFn(transformNodeObj);
	  MPlug transPlg = nodeFn.findPlug("translate"); // 引用的是translate属性， 包含三个子属性。

	  //子属性如何得到？ 是直接transPlg.child("translateX")? 不是。
	  MObject transXAttr = nodeFn.attribute("translateX");
	  MPlug transxPlg = transPlg.child(transXAttr); // 获取子接头
	  
	  获取子接头的父接头
	  transxPlg.parent();
	  
	  遍历子接头numChildren(), child(i)
	  
  }
  数组接头
  {
	  引用数组属性的接头称为数组接头，引用数组中元数的接头称为元素接头。
	  
	  unsigned int nElems = arrayPlug.numElements();
	  for(unsigned int i = 0; i < nElems; i++){
	  	MPlug elemPlug = arrayPlug.elementByPhysicalIndex(i); // 相对地还有个logicalIndex()
  	  }
	  
	  向数组中添加元素
	  MObject myNodeObj = dgMod.createNode(MyNode::id);
	  MFnDependencyNode depFn(myNodeObj);
	  MPlug scoresPlug = depFn.findPlug("scores");
	  
	  数组中的元素都有两个索引：物理索引和逻辑索引。当从数组中删除一个元素时，某些元素的物理索引就可能会发生变化，但逻辑索引不会发生变化。MEL中只能获得数组中元素的逻辑索引。
	  
	  在创建属性之间的连接时，会涉及到元素的物理索引和逻辑索引。属性之间的连接是基于它们的逻辑索引来完成的。
	  
	  // 创建新元素
	  MPlug scorePlg;
	  scorePlg = scoresPlug.elementByLogicalIndex(0); // 创建0号逻辑元素并设值，如果0号逻辑索引已存在，则不创建。
	  scorePlg.setValue(46);
	  
	  // 获取逻辑索引的一个完整列表
	  MIntArray logIndices;
	  scoresPlg.getExistingArrayAttributeIndices(logIndices);
	  
	  // 已知元素接头，获取父数组接头。跟复合接头不一样。
	  MPlug arrayPlg = scorePlg.array();
  }
  ```
  - 七、数据块datablock
    - 数据块是真正存储节点所需数据的地方。节点除了在一个或多个数据块中存储数据外，不会在其他地方存储保存数据。 
    - 开发人员无需知道数据块内部的工作机制以及它们是如何一起协调工作的，而只需知道使用MDataBlock数据块，MDataHandle类和MArrayDataHandle类来提取和设置节点的数据就可以了。
  
  ```
  1.访问单值的节点属性数据
  data.inputValue(attributeName), data.outputValue(attributeName)。这两个函数都将返回一个MDataHandle类的实例来访问数据。
    // MDataHandle      inputValue ( const MPlug & plug, MStatus * ReturnStatus = NULL ); // 数据是只读的
    // MDataHandle      inputValue ( const MObject & attribute, MStatus * ReturnStatus = NULL ); // 只能用来改写
  2.访问数组属性时，应当使用data.inputArrayValue()和data.outputArrayValue()函数。这两个函数都会返回一个MArrayDataHandle。用它可以访问数组接头中的单个元素。
  {
  	MArrayDataHandle scoresHnd = data.inputArrayValue(scoresAttr);
	MdataHandle avgHnd = data.outputValue(avgAttr);
	const unsigned int nElems = scoresHnd.elementCount();
	for(i = 0; i < nElems; i++){
		scoresHnd.jumpToElement(i);
		MDataHandle elemHnd = scoresHnd.inputValue();
		sum += elemHnd.asInt();
	}
	sum /= nElems;
	avgHnd.set(sum);
	
	data.setClean(plug);
  }
  
  ```
  - 八、节点设计指导方针
    - 保持节点功能的单一性
    - 节点应始终不“知道”它所处的环境或语境
    - 节点不应使用其自身之外的数据。
    - 一个节点始终不应该检查自己的属性是否连接到其他节点。
    - 一个节点应始终不能确定自己何时被求值。
    
  ```
  
  ```
- 定位器
  - BasicLocator 
- 操纵器
  - 跟Attribute Editor一样是用来修改属性值的，但更直观。
- 变形器
- 高级C++ API
```
节点引用
MNodeMessage的addNameChangeCallback函数

代理对象
MPxNode类的userNode函数
postConstructor函数

网络化接头和非网络化接头
接头代表属性的引用：MPlug::info函数可以查看路径，记得查看数组接头
接头存储连接信息
为每个节点保存了一个接头树。 称为网络化接头。从逻辑上推理，节点所有被连接的属性都对应这一个网络化接头。
MPlug::isNetworked

语境
MDGContext
MDGContext::isNormal函数可判断是否是当前时刻
接头在某一时刻求值

数据块
MPxNode的forceCache函数可在给定时间为节点强制创建一个数据块。使用MDataBlock类的context函数来获得给定数据块的语境。

传播标记？？
传播标记可以有效地阻止已经被设置的接头的脏位标记消息的传播。dgdirty命令。

直接传递处理
nodeState属性定义了节点是否应该计算它的输出属性。是个枚举值， 0正常， 1直接传递，2阻塞，3内部禁止
直接传递：直接将输入属性作为输出属性输出。

循环的节点从属关系




``` 
# 其他

## 官方文档 http://help.autodesk.com/view/MAYAUL/2018/CHS/?guid=__Commands_index_html

## 常用一般命令
```
duplicate -upstreamNodes $inNodes;
listRelatives -parent nodename;
connectAttr
findType
objectType
pointPosition Louise_Anim_with_Marker_center:Louise_Anim_with_Marker_left:Louise_Anim_With_Marker:Louise.vtx[200];
Python cmds.pointPosition("Louise_Anim_with_Marker_center:Louise_Anim_with_Marker_left:Louise_Anim_With_Marker:Louise.vtx[116]");
getAttr  Louise_Anim_with_Marker_center:Louise_Anim_with_Marker_left:Louise_Anim_With_Marker:Louise.vtx[200]
```

## Maya SDK
博客 https://blog.csdn.net/whwst/article/details/81604853
https://www.autodesk.com/developer-network/platform-technologies/maya


## Maya devkit源代码阅读
```
asciiToBinary.cpp
#include <maya/MStatus.h>
#include <maya/MString.h> 
#include <maya/MFileIO.h>
#include <maya/MLibrary.h>
#include <maya/MIOStream.h>

helloWorld.cpp
#include <maya/MGlobal.h>

MStatus::perror(...);
MGlobal::displayInfo("Hello world! (script output)" );
MGlobal::displayError
MGlobal::displayWarning

MGlobal::executeCommand( "print \"Hello world! (command script output)\\n\"", true );

readAndWrite.cpp
MFileIO::open()
MFileIO::saveAs()
MFileIO::exportAll
MFileIO::currentFile()

surfaceCreate.cpp
#include <maya/MObject.h>
#include <maya/MDoubleArray.h>
#include <maya/MPointArray.h>
#include <maya/MPoint.h>
#include <maya/MFnNurbsSurface.h>
MFnNurbsSurface mfnNurbsSurf;
mfnNurbsSurf.create(...)

surfaceTwise.cpp
#include <maya/MVector.h>
#include <maya/MMatrix.h>
#include <maya/MArgList.h>
#include <maya/MSelectionList.h>
#include <maya/MItSelectionList.h>
#include <maya/MItSurfaceCV.h>
#include <maya/MItMeshVertex.h>
#include <maya/MDagPath.h>
MGlobal::selectByName(surface1, MGlobal::kReplaceList);

MSelectionList slist;
MGlobal::getActiveSelectionList( slist );
if (iter.isDone()){}
status = iter.getDagPath( objectPath, component );
MFn::kNurbsSurface, MFn::kMesh, M
MS == MStatus
MS::kFailure;
MItMeshVertex vertIter( objectPath, component, &status );
for ( ; !vertIter.isDone(); vertIter.next() ) {}
MPoint pnt = vertIter.position( MSpace::kWorld );
status = vertIter.setPosition( pnt, MSpace::kWorld );
vertIter.updateSurface(); // Tell maya to redraw the surface with all of our changes
MItSurfaceCV cvIter( objectPath/*MDagPath*/, component/*MObject*/, true, &status );

// Stream Indexing System
adskDataIndex.h
adskDataChannel.h
adskDataStream.h


```


- http://help.autodesk.com/view/MAYAUL/2018/CHS/?guid=__cpp_ref_index_html
- github: https://github.com/topics/maya-plug
- open source Maya rigging and animation software https://github.com/mgear-dev/mgear_dist

## Alembic
Alembic是一个开源的CG通用格式。 Alembic将复杂的动画场景提取为一组非程序化的，与应用程序无关的烘焙几何体结果。



## Maya知识

- 常规编辑器->显示层编辑器 可以控制可见性及播放时是否可见 
- BlendShape相关
  - 窗口->动画编辑器->形变编辑器
    - 文件->导出形状
  
  - 窗口->动画编辑器->姿势编辑器
  
  - 动画模式
    - 播放->播放预览 playblast 
    ```
    //录制动画
    playblast -startTime $start -endTime $end  -format video -filename $output_file -forceOverwrite -sequenceTime 0 -clearCache 1 -viewer 1 -showOrnaments 0 -offScreen  -fp 4 -percent 100 -compression "IYUV 编码解码器" -quality 70 -widthHeight $width $height;
    
    //写文件
    $field_camera = `fopen $output_cameras_txt_file "w"`;
    fprint $field_camera $camera_model;
    fclose $field_camera;
    ```
  
  - 已经有blendshape的shape上如何添加sculpt。
    - 首先再添加一个blendshape，并添加一个目标为当前目标，就可以再进行雕刻了。
    
- 在纹理上绘制
渲染->纹理->3D绘制工具

- ncloth

- 贴图
  - 法线贴图 http://download.autodesk.com/us/maya/maya_2014_gettingstarted_chs/index.html?url=files/GUID-267CCDE6-5697-4235-9728-805879FEBA2A.htm,topicNumber=d30e25303


## FBXSDK
- http://help.autodesk.com/view/FBX/2019/ENU/
- Fbx动画 http://help.autodesk.com/view/FBX/2019/ENU/?guid=FBX_Developer_Help_animation_animation_data_structures_animation_classes_interrelation_html
- object model
  - objects, properties, connections, error handling
- scene
  - transformations, global and local transform
    - lNode->EvaluateGlobalTransformation()
    - lNode->EvaluateLocalTransformtaion()
    - equal to use FbxAnimEvaluator
    - lEvaluator->GetNodeGlobalTransform(lNode) and lEvaluator->GetNodeLocalTransform(lNode)
  - 动画节点的动画数据 FbxTime lTime;
    - lTime.SetSecondDouble((float)2);
    - FbxAMatrix &lGlobalTransform = lNode->EvaluateGlobalTransform(lTime);
- node 
  - transform
  - attribute
- node properties
  - FbxGeometry, FbxLight, FbxCamera, FbxMarker， FbxMesh
- 动画 animation http://help.autodesk.com/view/FBX/2019/ENU/?guid=FBX_Developer_Help_animation_animation_data_structures_animation_classes_interrelation_html
  - 数据结构 FbxAnimStack, FbxAnimLayer, FbxAnimCurveNode, FbxAnimCurve, FbxAnimCurveKey
  - FbxScene可以有0个或者多个FbxAnimStack。场景没有动画就不需要用Stack.
  - FbxAnimStack是最高级别的动画数据容器。可以包含一个或者多个FbxAnimLayer.
  - FbxAnimLayer包含一个或者多个FbxAnimCurveNode， 这些node连接到FbxAnimCurves。
  - FbxAnimCurve代表函数，可以驱动property的动画
  
## UI写法
```
用符号代表CheckBox.
window;
columnLayout;
    symbolCheckBox -image "circle.png";
    symbolCheckBox -image "sphere.png";
    symbolCheckBox -image "cube.png";
showWindow;

gridLayout有的时候非常有用，比如写简单几个按钮时，如果想把按钮调整好位置，可以用gridLayout.

// 带标题的表格。
window;
formLayout form;
componentBox cbox;
formLayout -e
    -af cbox "top" 0
    -af cbox "left" 0
    -af cbox "right" 0
    -af cbox "bottom" 0
    form;
showWindow;

// 这个例子非常好。
string $window = `window`;
string $form = `formLayout -numberOfDivisions 100`;
string $b1 = `button`;
string $b2 = `button`;
string $column = `columnLayout -adjustableColumn true`;
button; button; button;

formLayout -edit
    -attachForm     $b1     "top"    5
    -attachForm     $b1     "left"   5
    -attachControl  $b1     "bottom" 5 $b2
    -attachPosition $b1     "right"  5 75

    -attachNone     $b2     "top"
    -attachForm     $b2     "left"   5
    -attachForm     $b2     "bottom" 5
    -attachForm     $b2     "right"  5

    -attachForm     $column "top"    5
    -attachPosition $column "left"   0 75
    -attachControl  $column "bottom" 5 $b2
    -attachForm     $column "right"  5
$form;

showWindow $window;
```


