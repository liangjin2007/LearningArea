# Maya5编程全攻略

网格显示->切换
## Maya架构
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
  
## Maya MEL
类c语言
变量：类型，数组，vector，boolean， 字符串， int, float
操作符：关系，逻辑，算数
控制:条件，循环
脚本：存储脚本，自定义脚本目录 MAYA_SCRIPT_PATH环境变量

模式:创建，查询，编辑

获取顶点世界坐标位置： ??? 获取的顶点位置不对

- 常用MEL命令
```
ls
ls -sl
ls -type
whatIs // 返回mel语言类型
pointPosition
getAttr
setAttr
动态属性addAttr -longName "points" -attributeType int 5;
检查属性是否存在attributeExists("points", object)
使得属性变得可关键帧化setAttr -keyable
删除动态属性 deleteAttr
重命名动态属性renameAttr
显示属性信息 listAttr -userDefined, listAttr -keyable
getAttr -type nurbsSphere1.points; // Result : long
attributeQuery

- 动画
currentUnit -query -time;
currentTime 10;
getAttr -time 10 sphere.traslateX; // 这边可以优化下我之前的代码。
play;
playbackOptions xxx;
playblast -file xxx.mov;
- 动画曲线

- 关键帧
keyframe -query -name ball; // 返回节点的动画曲线节点
keyframe -query -name ball.translateX; // 返回属性的动画曲线节点
isAnimCurve() 是否是动画曲线
listAnimatable -type ball // 要确定一个节点是否是可动画化的。

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
window
showWindow
columnLayout myLayout;
button
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
  
