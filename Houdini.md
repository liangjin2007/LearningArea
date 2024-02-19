## Learning Video https://www.youtube.com/watch?v=NwabG-znu9Y
## HDK https://www.sidefx.com/docs/hdk/index.html
- Introduction to the HDK
这边只关心Windows平台上的开发
```
1. 需要用对应于Houdini的编译器版本 vc142 Microsoft Visual C++ 2019, version 16.9.4(Windows SDK version?)
2. 需要MSVCDir环境变量指定VC subdirectory.
3. HDK应用例子：standalone/geoisosurface.C， 一个包含main函数的应用，生成一个后缀为.bgeo的几何文件。 
3.1. 命令行编译：Start->All Programs->Side Effects Software -> Houdini X.Y.ZZZ ->Command Line Tools

  mkdir HDK
  cd HDK
  copy "C:\Program Files\Side Effects Software\Houdini X.Y.ZZZ\toolkit\samples\standalone\geoisosurface.C"
  hcustom -s geoisosurface.C // 这个会生成一个./geoisosurface.exe应用程序。

3.2. 执行
  ./geoisosurface.exe // 会生成sphere.bgeo
  gplay sphere.bego // 用gplay应用查看这个文件。

```

- 插件开发
```
1. 实际是一个dll
2. Houdini启动的时候会加载
3. 可以包含定制的node operators, script commands 及其他对象
4. 插件加载顺序
5. 定位和执行dll中的hook函数（注册HDK代码）
6. cmd.exe, -> hscript -> dsoinfo命令用于看看加载了哪些插件。
7. HOUDINI_DSO_PATH
```


## 如何开发刷子
### 1. 看完 HDK -> Houdini Operators
- Architectural Overview
```
Node Organization
  Houdini scene : 一个节点层次组成。 每个节点有一个Path。 OP_Node, OP_Network。 /obj/geo1是一个对象， 类型是OP_Network, 包含一个SOP_Node 类型对象的集合。
  / 即 Director。 /中的nodes有的时候称为Managers。
  Managers: /obj, /ch, /out, /shop, /img, /vex
OP_Director
  OPgetDirector()->clearPickedItems();
Class Hierarchy
```

- Working with Nodes
```
Paths
Create Nodes
Traversing Nodes
Flags
Selections
  node->pickRequest(bool clear);
  OPgetDirector()->getPickedNodes(picked_nodes);
Grouping Nodes in Bundles
  OPgetDirector()->getBundles();
Casting Nodes
  OBJ_NODE* obj_node = CAST_OBJNODE(node);
  sop_node = CAST_SOPNODE(obj_node->getChild("file1"));
Cooking Nodes
  cookMe()
```  

- Working with Parameters
```
Basics
  OP_Parameters contains a PRM_ParmList, which owns an array of PRM_Parm objects. 参数有component, 每个component有一个可选的channel（CH_CHannel对象）
  parameter evaluation functions
  CHgetEvalTime()
  OP_Context::getTime()
  float tx = node->evalFloat("t", component_index, context.getTime());
Multi-Params
Ramp Parameters
```

- Thread Safety
```

```

- Operator Contexts
```
OBJ Concepts
SOP Concepts
  creating and modifying Houdini geometry.
  Cooking SOPs
    SOP_Node::cookMySop()
    SOP_Node::duplicateSource, etc
  Gorups in SOPs
    SOP_Node::parseGroups
    SOP_Node::parsePrimitiveGorups
    SOP_Node::parsePointGroups
  Node Flags
    OP_Network::getDisplayNodePtr() and OP_Network::getRenderNodePtr()
    setBypass
    setLock
  Buiding Custom SOPs
  Retrieving Cooked Data
    void accessGeometry(SOP_Node* sop_node, fpreal cook_time)
    {
        // Get our parent.
        OP_Node *parent_node = sop_node->getParent();
        
        // Store the cooking status of our parent node.
        bool was_cooking = false;
        if(parent_node)
        {
            was_cooking = parent_node->isCookingRender();
            parent_node->setCookingRender(true);
        }
        // Create a context with the time we want the geometry at.
        OP_Context  context(cook_time);
        // Get a handle to the geometry.
        GU_DetailHandle gd_handle = sop_node->getCookedGeoHandle(context);
        // Check if we have a valid detail handle.
        if(!gd_handle.isNull())
        {
            // Finally, get at the actual GU_Detail.
            const GU_Detail* gdp = gd_handle.gdp();
            // Do something with our geometry.
            // ...
        }
        // Restore the cooking flag, if we changed it.
        if(parent_node)
            parent_node->setCookingRender(was_cooking);
    }
  Transformations
    bool getWorldTransform(SOP_Node* sop_node, fpreal cook_time, UT_DMatrix4 &xform)
    {
        OP_Context  context(cook_time);
        OBJ_Node *  creator = CAST_OBJNODE(sop_node->getCreator());
        return creator->getLocalToWorldTransform(context, xform);
    }
```
-Building Custom Operators
```

```




## Samples positions Side Effects Software\Houdini 18.5.696\toolkit
## Hierarchical Set of Library
```
Utility Libraries
  AU
  CL
  CVEX
  EXPR
  IMG/TIL
  VEX
  SYS/UT - Basic Utilities
  Third Party Dependencies: Boost, Intel TBB, etc
Operator Libraries
  CH - Channel(animation) Library
  PRM - Parameters
  OP - Base classes for all Houdini nodes
    Operator definitions
    Node interfaces
    Galleries - parameter presets
    Takes - layer of parameters
  Specialized Operator Libraries
    OBJ - Objects, SOP - Surface Ops, CHOP - Channels Ops, etc
    Mirror of Houdini node contexts
Node Architecture
  // UML 图示符号
Node Libraries
Geometry Libraries
Simulation/DOPs
Specialized Node Libraries
Some UI
Python Customnization
```


## Nodes
https://www.sidefx.com/docs/houdini/nodes/index.html



## C++ API HDK
https://www.sidefx.com/docs/hdk/index.html

### 介绍
- hello world
```
0001: #include <GU/GU_Detail.h>
0002: 
0003: static float
0004: densityFunction(const UT_Vector3 &P)
0005: {
0006:     return 1 - P.length(); // Return signed distance to unit sphere
0007: }
0008:
0009: int
0010: main(int argc, char *argv[])
0011: {
0012:      GU_Detail            gdp;
0013:      UT_BoundingBox       bounds;
0014:
0015:      bounds.setBounds(-1, -1, -1, 1, 1, 1);
0016:      gdp.polyIsoSurface(HDK_Sample::densityFunction, bounds, 20, 20, 20);
0017:      gdp.save("sphere.bgeo", true, NULL);
0018:
0019:      return 0;
0020: }
```
```
编译
  hcustom -s geoisosurface.C
运行
  ./geoisosurface
查看结果  Houdini有个gplay应用程序
  gplay sphere.bgeo 

```


### Houdini Operators
- 架构
```
节点组织：
  Node = path /obj/geo1/file1， OP_Node
  Network = each of the "directory levels", 用类OP_Network表示
  Director = Root network / , 用类OP_Director或者MOT_Director



Object Manager /obj
  contain OBJ_Node类型的Objects, object contain surface operators SOP SOP_Node. 
  也contain OBJ_DopNet, DOP operators在OBJ_DopNet节点之下
Channel Manager /ch, 应该是处理关键帧的。
  CHOPNet_Node, CHOP_Node
Output Manager /out
  nodes derived from ROP_Node
Shader Manager /shop
  SHOP_Node
Composite Manager /img
  COPNet_Node, COP2_Node
Shader Code Manager /vex
  VOPNet_Node, VOP_Node




Parameters
  OP_Parameters是OP_Node的父类， 它包含一个PRM_ParmList对象， 拥有一个PRM_Parm object数组
  float now = context.getTime();
  node->evalFloat("t", 0/*component index*/, now);
  node->setFloat ...
  node->setChRefFloat


动态参数 Multi-Params, 每个子实例由PRM_Template定义。

Ramp Parameters
OP_Context

```


- Working with Nodes
```
Paths : UT_String 类型

node to path : node->getFullPath(path)
node to relative path
  
  
path to node pointer
    绝对路径 OPgetDirector()->findNode(path)
    相对路径
    OPgetDirector()->findObjNode(path)
    OPgetDirector()->findSOPNode(path)

Creating nodes
    获取时间： float t = context.getTime();
    node = parentOPNetwork->createNode(nodetype, nodeName);
    node->runCreateScript();
    node->setFloat("t", 0, t, 0.f);
    OP_Node* input = parentOPNetwork->findNode("null1");
    node->setInput(0, input); //设置第一个输入
    node->moveToGoodPosition(); // 
    
Traversing Nodes
    parent->traverseChildren(nodeCallback, (void*)prefix, true);
    
Flags
    Expose
    ByPass
    Template : as reference geometry that is not selectable
    Footprint: viewport as reference geometry that is not selectable
    Highlight : in the viewport
    Display : displayed geometry for the network
    Render : as the main rendering geometry
    Pickable: pickable from viewport
    Xray: displayed as Xray geometry which is drawn in wireframe
    Audio
    Export
    
Selections
    node1->pickRequest(true); // clear node selection and then select node1
    node2->pickRequest(false); // add node2 to the selection
    OPgetDirector()->clearPickedItems();
    OPgetDirector()->getPickedNodes(pickedNodes); // OP_NodeList
    OPgetDirector()->getLastPickedNode();
    parent->getCurrentNodePtr();
    
Grouping Nodes in Bundles
    OP_BundleList * list = OPgetDirector()->getBundles();
    list->createBundles
    list->getBundle("bundlename");
    list->bundleAddOps(bundle, nodes);
    list->bundleRemoveOps(bundle, nodes);
    bundle->getMembers()
    
Globbing: 是通配符的意思，用来干搜索/匹配的事情
    mgr = OPgetDirector()->getManager("/obj");
    bundle = OPgetDirector()->getBundles()->getPattern(
      bundle_name,
      mgr,/*creator*/
      mgr,/*relative_to*/
      "mynode*",
      "!!OBJ!!" // only accept object node
    ); // 内部会创建一个名为bundle_name的bundle
    
    bundle->getMembers(nodes);
    
    OPgetDirector()->getBundles()->deReferenceBundle(bundle);
    
Parameter Bundles
     
Casting Nodes
    CAST_OBJNODE(node)
    CAST_SOPNODE(obj_node->getChild("file1"));
Cooking Nodes

```


- Working with Parameters
```
Basics
  OP_Parameters <- OP_Node
    has OP_ParamList
      OP_Param can have a number of components. A component has an optional channel
    得到参数 evalxxx函数
    设置参数 setxxx函数
  Multi-Params
     float   t = context.getTime();
    int     num_instances = blend_sop->evalInt("nblends", 0, t);
    int     start_idx = blend_sop->getParm("nblends").getMultiStartOffset();
    int     instance_idx;
    // evaluate all multi-parm instances
    float * weights = new float[num_instances];
    for (int i = 0; i < num_instances; i++)
    {
        instance_idx = start_idx + i;
        weights[i] = evalFloatInst("weight#", &instance_idx, 0, t);
    }
    // ... do something with weights ...
    delete [] weights;
    // add a new multi-parm instance with a 0.2 weight value
    blend_sop->setInt("nblends", 0, t, num_instances + 1);
    instance_idx = start_idx + num_instances;
    blend_sop->setIntInst(0.2, "weight#", &instance_idx, 0, t);
    
  Ramp Parameters
    UT_Ramp
    node->updateRampFromMultiParm(t, node->getParm("ramp_name"), ramp);
    ramp.rampLookup(u, values);
    node->updateMultiParmFromRamp(t, ramp, node->getParm("ramp_name"), false);
```

- Thread Safety
```
Modifying parameters or node data while evaluting them
Adding/Removing points, attributes, primitives, on a GU_Detail while another thread access the same GU_Detail.
SIM_DATA_GET in DOPs
```

- Operator Contexts
```
OBJ Concepts
SOP Concepts
  SOP_Node
  SOP_Network
DOP Object 
  SIM_Object
```

# Houdini python node
- https://zhuanlan.zhihu.com/p/363920596
- hou API https://www.sidefx.com/docs/houdini/hom/hou/index.html
- hou.Node class https://www.sidefx.com/docs/houdini/hom/hou/Node.html
- hou.Geometry class https://www.sidefx.com/docs/houdini/hom/hou/Geometry.html
```
node = hou.pwd()
geo = node.geometry()

# Add code to modify contents of geo.
# Use drop down menu to select examples.
points = geo.points()
prims = geo.prims()

# add attribute
a = geo.addAttrib(hou.attribType.Point, "a", (1.0, 0.0, 0.0))
b = geo.addAttrib(hou.attribType.Prim, "b", (1.0, 0.0, 0.0))


aStr = hou.readFile("D:/1.txt").split(" ")
bStr = hou.readFile("D:/2.txt").split(" ")

for i in range(len(points)):
    p = points[i]
    avalue = float(aStr[i])
    if avalue > 0.9:
        avalue = 1.0
    else:
        avalue = 0.0
    #print("avalue %r" % avalue) 
    p.setAttribValue(Color, (avalue, 0.0, 0.0))
    
b_len = len(bStr)

print("b_len %r prim_len %r" % (b_len, len(prims)))

for i in range(len(prims)):  
    p = prims[i]
    bvalue = bStr[i]
    
    #print("bvalue %r" % bvalue) 
    c = 1.0
    if bvalue == 5:
        c = 1.0
    else:
        c = 0.0
    
    p.setAttribValue(b, (c, 0.0, 0.0))

```

# Houdini nodes
```
File
Alembic
Unpack
Group Create
Python
Visualize
Divide
```



