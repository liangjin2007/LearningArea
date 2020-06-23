# Maya5编程全攻略

网格显示->切换
## Maya架构
- 数据流模型 DG， 技术上讲基于一种推拉模型，而不是基于严格的数据流模型。
- 数据及其操作被封装为节点， 一个节点含有任意数目的插槽，其中含有Maya使用的数据，节点也包含一个操作符，用于对其数据进行处理。
- 场景就是DG
- Hypergraph
- 节点类型
```
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
```
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
string $nodes[]=`keyframe -q -nname ball.translateX`;
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
切线性质： 类型，角度，加权和锁定





```
### 图形用户界面
```
window
showWindow
columnLayout myLayout;
button
```
### 表达式

- 常用MEL命令
```
pointPosition

- 关键点
创建/编辑/查询关键点
setKeyframe
setDrivenKeyframe
keyTangent
copyKey
cutKey

- 骨架
insertJoint
outputJoints脚本
string $inPlugs[] = `listConnections -plugs yes -destination yes node.attr`// page 143
string $tokens[];
tokenize $inPlugs[0] "." $tokens；
string $inNode = $tokens[0];
string $inAttr = $tokens[1];
duplicate -upstreamNodes $inNodes;
listRelatives -parent nodename;
connectAttr

- 运动路径



findType
objectType
pointPosition Louise_Anim_with_Marker_center:Louise_Anim_with_Marker_left:Louise_Anim_With_Marker:Louise.vtx[200];
Python cmds.pointPosition("Louise_Anim_with_Marker_center:Louise_Anim_with_Marker_left:Louise_Anim_With_Marker:Louise.vtx[116]");
获取局部坐标位置
getAttr  Louise_Anim_with_Marker_center:Louise_Anim_with_Marker_left:Louise_Anim_With_Marker:Louise.vtx[200]
findType
group
move
parent
inheritTransform -ff xxx;
scale
rotate
xform
```

## Maya图形用户界面 ~page150

简单

## Maya SDK
博客 https://blog.csdn.net/whwst/article/details/81604853
https://www.autodesk.com/developer-network/platform-technologies/maya

## Maya C++ API (page 208)
```
此API可修改的领域和功能
命令，DG节点，工具，文件转换器，变形器，着色器，操纵器，定位器，动力场，发射器，形体，解算器
- API 一般性
M, Maya类
MPx 代理对象 MPxNode
MIt 迭代类 MItDag, MItMeshEdge
MFn 函数集 MFnMesh, MFnDagNode 
Maya方法与普通面向对象方法的区别： 数据与函数分来。 MFn只包含函数。
MObject所有数据均是通过类MObject访问的。实际指向Maya核心内一个句柄。
Maya拥有所有的数据，而你不拥有它内部的任何数据。
MFn函数集用来创建、编辑和删除数据。 比如要创建transform节点，可以这样。
MFnTransform transformFn;
MObject transformObj = transformFn.create();
MFnBase -> MFnDependencyNode -> MFnDagNode -> MFnTransform
- 开发插件
错误，报告MStatus， 集成。加载/卸载pluginInfo -q xxx;
unloadPlugin full_file_path;
loadPlugin full_file_path;
MAYA_PLUG_IN_PATH

MString, MObject, MSelectionList, MStatus
- 插件代码
  - 命令
    - 一个简单修改动画关键点属性值的命令
    - 添加命令参数
    - 提供帮助
    - Undo/Redo 可撤销和不可撤销
    - 编辑和查询
  - 节点
    - MDGModifier用于创建节点以及必要的连接。
    - MDataHandle
```

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
MGlobal::displayInfo("Hello world! (script output)" );
MGlobal::executeCommand( "print \"Hello world! (command script output)\\n\"", true );

readAndWrite.cpp
MStatus::perror(...);
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
  
