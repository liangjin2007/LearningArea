## Learning Video https://www.youtube.com/watch?v=NwabG-znu9Y

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
