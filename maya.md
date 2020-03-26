# Maya5编程全攻略

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
    - unitConversion
    - dagPose ~~bindPose~~
    - blendWeighted ~~BW_blendShape1_Eyebrows_Frown_R~~
    - lambert
    - dx11Shader
    - file
    - place2dTexture
    - surfaceShader
    - shadingEngine
 
   
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
节点：

获取顶点世界坐标位置： ??? 获取的顶点位置不对
MEL
pointPosition Louise_Anim_with_Marker_center:Louise_Anim_with_Marker_left:Louise_Anim_With_Marker:Louise.vtx[200];
Python cmds.pointPosition("Louise_Anim_with_Marker_center:Louise_Anim_with_Marker_left:Louise_Anim_With_Marker:Louise.vtx[116]");
获取局部坐标位置
getAttr  Louise_Anim_with_Marker_center:Louise_Anim_with_Marker_left:Louise_Anim_With_Marker:Louise.vtx[200]


- findType
```
findType [-deep] [-exact] [-forward] [-type string]

findType is NOT undoable, NOT queryable, and NOT editable.

The findType command is used to search through a dependency subgraph on a certain node to find all nodes of the given type. The search can go either upstream (input connections) or downstream (output connections). The plug/attribute dependencies are not taken into account when searching for matching nodes, only the connections.
Return value
string[]	The list of node(s) of the requested type connected to the given node(s)

```

## Maya SDK
博客 https://blog.csdn.net/whwst/article/details/81604853
https://www.autodesk.com/developer-network/platform-technologies/maya


## Maya C++ API
- http://help.autodesk.com/view/MAYAUL/2018/CHS/?guid=__cpp_ref_index_html
- github: https://github.com/topics/maya-plug
- open source Maya rigging and animation software https://github.com/mgear-dev/mgear_dist

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
    


