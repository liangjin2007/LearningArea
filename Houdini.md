## Learning Video https://www.youtube.com/watch?v=NwabG-znu9Y
## HDK https://www.sidefx.com/docs/hdk/index.html
### 1. 看完 Introduction to the HDK
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

### 2. 看完 Houdini Operators
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
Registration
  inputs数目
  是否加入TAB菜单

Parameters
  Basics
  Parameter Types
  SWitchers
  Multi-Parms
  Ramps
  Menus
  Disabling and Hiding Parameters
    OP_Parameters::updateParmsFlags
  Obsolete Parameters
    OP_Operator::setObsoleteTemplates

Cooking
  Handling Interrupts
  #include <UT/UT_Interrupt.h>
  {
      // Interrupt scope closes automatically when 'progress' is destructed.
      UT_AutoInterrupt progress("Performing My Operation");
      // some loop ... using SOPs as an example
      for (GA_Iterator it(gdp->getPointRange()); !it.atEnd(); ++it)
      {
          // test if user requested abort
          if (progress.wasInterrupted())
              break;
      }
  }

Local Variables
  // First, let us get a bunch of unique integers
  enum {
      VAR_PT,     // Current point
      VAR_NPT,    // Total number of points
      VAR_ID,     // Particle ID
      VAR_LIFE,   // Life time
  };
  // Next, we define our local variable table
  CH_LocalVariable
  SOP_MyNode::myVariables[] = {
      { "PT",     VAR_PT },
      { "NPT",    VAR_NPT },
      { "ID",     VAR_ID },
      { "LIFE",   VAR_LIFE },
      { 0 }       // End the table with a null entry
  };
  bool
  SOP_MyNode::evalVariableValue(fpreal &val, int index, int thread)
  {
      GA_Offset        ppt_offset;
      int              id;
      UT_Vector2D      life;
      if (!GAisValid(myCurrPoint))    // Sorry, we're in an invalid state
          return false;
      switch (index)
      {
          case VAR_PT:
              val = fpreal(myCurrPoint);
              return true;
          case VAR_NPT:
              val = fpreal(myTotalPoints);
              return true;
          case VAR_ID:
              if (myInputGeo && myInputIdHandle.isValid())
              {
                  ppt_offset = myInputGeo->pointOffset(myCurrPoint);
                  id = myInputIdHandle(ppt_offset);
                  val = fpreal(id);
                  return true;
              }
              return false;
          case VAR_LIFE:
              if (myInputGeo && myInputLifeHandle.isValid())
              {
                  ppt_offset = myInputGeo->pointOffset(myCurrPoint);
                  life = myInputLifeHandle(ppt_offset);
                  val = life.x() / life.y();
                  return true;
              }
              return false;
      }
      // Not one of our variables, must delegate to the base class.
      return SOP_Node::evalVariableValue(val, index, thread);
  }
  OP_ERROR
  SOP_MyNode::cookMySop(OP_Context &context)
  {
      fpreal  t = context.getTime();
      ...
      // Now, we loop through each point, doing expression
      // evaluation INSIDE the loop.
      for (myCurrPoint = GA_Index(0); myCurrPoint < myTotalPoints; myCurrPoint++)
      {
          // When evaluating parms, it's possible that the expression
          // would use one of our local variables, so 
          pos.x() = evalFloat("posx", 0, t);
          ...
      }
      myCurrPoint = GA_INVALID_INDEX; // Make sure to flag that we're done cooking
      ...
  }

Dependencies
  Name Dependencies
    比如参数为其他node名，那么当这个node名字改变时需要相应的改变
    PRM_TYPE_DYNAMIC_PATH
  Data Dependencies
    data dependency graph
    OP_Node::addExtraInput(), 比如Object Merge SOP

    OP_ERROR 
    SOP_MyNode::cookMySop(OP_Context &context)
    {
        fpreal      t = context.getTime();
        UT_DMatrix4 xform(1.0); // identity
        OBJ_Node *  obj_node;
        UT_String   obj_path;
        // ...
        evalString(obj_path, "objpath", 0, t);
        obj_node = findOBJNode(obj_path);
        if (obj_node)
        {
            obj_node->getLocalToWorldTransform(context, xform); // use its data
            addExtraInput(obj_node, OP_INTEREST_DATA);          // tell Houdini
        }
        // ...
    }

    // If depend on other node's parameter.
    fpreal t = context.getTime();
    int    pi = other_node->getParmList()->getParmIndex("r");
    int    vi = 1;
    fpreal ry = other_node->evalFloat(pi, vi, t);
    addExtraInput(other_node, pi, vi);

Help
  overload OP_Operator::getHDKHelp
  text file operator_name.txt， 放到$HOME/houdiniX.Y/help/nodes/OPTYPE
  HTML server bool OP_Operator::getOpHelpURL(UT_String &str){ str.harden("http://xxxx"); return true; }

Icon
  svg format
  $HOME/houdiniX.Y/config/Icons

Writing a SOP
  Creating Geometry
    generator node
    SOP_Node::cookMySop
      local member variable called gdp（GU_Detail）
      GU_Detail::clearAndDestroy(), not fast
      GEO_Detail::stashAll(), GEO_Detail::destroyStashed()

      GU_Detail::appendPoint, GU_PrimPoly::build, 例子 SOP_Star.C

  Changing Geometry
    filter node
    OP_Node::lockInput
    SOP_Node::inputGeo
    OP_Node::unlockInput
    OP_Node::lockInputs and OP_Node::unlockInputs.
    SOP_Node::duplicateSource
    SOP_Node::duplicatePointSource
    SOP_PointWave.C
    SOP_HOMWave.C
    SOP_TimeCompare.C 可以看看怎么处理多个input。

  Local Variables in SOPs
    SOP_Node::evalVariableValue
    SOP_Node::setVariableOrder
    SOP_Node::setCurGdh
    SOP_Node::myGdpHandle
    SOP_Node::inputGeoHandle
    SOP_Node::setupLocalVars
    Cook的时候估计参数， OP_Node::evalInt。SOP_Node::myCurPtOff, SOP_Node::myCurVtxOff, SOP_Node::myCurVtxNum, and SOP_Node::myCurPrimOff.
    SOP_Node::resetLocalVarRefs.

  Adding Guide Geometry to SOPs
    SOPs 可以有2个regular guides(GOP_Guide)
    Cook guide: SOP_NodeFlags::setNeedGuide1 和  SOP_NodeFlags::setNeedGuide2
    SOP_Node::cookMyGuide1 和 SOP_Node::cookMyGuide2
    SOP_Flatten.C

  Manipulating Attributes inside SOPs
    SOP/SOP_TimeCompare.C
    SOP/SOP_DetailAttrib.C 

  Abusing SOP Cook Mechanism
  The Many Ways to Create a SOP
```



## CMake 模板
```
cmake_minimum_required( VERSION 3.20 )

project( xxx )
set(CMAKE_CXX_STANDARD 17)

# Assume one element of $ENV{PATH} is HOUDINI bin path.
# get houdini path to VAR houdini_path
set(houdini_path "")
string(REPLACE "\;" ";" PATH_LIST "$ENV{PATH}") # remove the last slash
string(REPLACE "\n" "" PATH_LIST "${PATH_LIST}") # remove the \n
foreach(p ${PATH_LIST})
	message(warning ${p})
	if(EXISTS "${p}/houdini.exe")
		cmake_path(GET p PARENT_PATH arg0)
		set(houdini_path ${arg0})
		message(warning "Houdini Path = ${houdini_path}")
	endif()
endforeach()
if(NOT EXISTS ${houdini_path})
	message(FATAL_ERROR "Houdini path is not found from environment PATH = $ENV{PATH}.")
endif()

message(STATUS "Houdini Path ${houdini_path}")


# CMAKE_PREFIX_PATH must contain the path to the toolkit/cmake subdirectory of
# the Houdini installation. See the "Compiling with CMake" section of the HDK
# documentation for more details, which describes several options for
# specifying this path.
list( APPEND CMAKE_PREFIX_PATH "${houdini_path}/toolkit/cmake" )

# Locate Houdini's libraries and header files.
# Registers an imported library target named 'Houdini'.
find_package( Houdini REQUIRED )

set( library_name xxx )

# add some extra include and library directories.
include_directories("${CMAKE_CURRENT_LIST_DIR}/../xxx")
link_directories("${CMAKE_CURRENT_LIST_DIR}/../xxx")

# Add a library and its source files.
file(GLOB TOOL_C_SOURCES "*.C")
file(GLOB TOOL_CPP_SOURCES "*.cpp")
file(GLOB TOOL_UIS "*.ui")
file(GLOB TOOL_HEADERS "*.h")
file(GLOB TOOL_ICONS "*.svg")
add_library( ${library_name} SHARED ${TOOL_C_SOURCES} ${TOOL_CPP_SOURCES} ${TOOL_HEADERS} ${TOOL_UIS} ${TOOL_ICONS})

# Link against the Houdini libraries, and add required include directories and
# compile definitions.
target_link_libraries( ${library_name} Houdini <extra_xxx_library>)
target_include_directories( ${library_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}) # for xxx.proto.h

# Sets several common target properties, such as the library's output directory.
houdini_configure_target( ${library_name} )

houdini_get_default_install_dir( H_OUTPUT_INSTDIR ) # get C:/users/xxxx/Documents/HoudiniX.Y/xxx

message(STATUS "Output Dir ${H_OUTPUT_INSTDIR}")
message(STATUS "Houdini Package Version ${Houdini_VERSION}")

foreach( f ${TOOL_ICONS} )
file(COPY "${f}" DESTINATION "${H_OUTPUT_INSTDIR}/config/Icons/")
endforeach()

foreach( f ${TOOL_UIS} )
file(COPY "${f}" DESTINATION "${H_OUTPUT_INSTDIR}/config/Applications/")
endforeach()

message(STATUS "C Files: ${TOOL_C_SOURCES}")
message(STATUS "CPP Files: ${TOOL_CPP_SOURCES}")

houdini_generate_proto_headers(OUTPUT_VAR PROTO_HEADERS FILES ${TOOL_C_SOURCES}) # generate propo.h for all .C files.
message(STATUS "PROTO_HEADERS ${PROTO_HEADERS}")
```


## 看HDK C++ 代码
### 理解各种概念
可以搜索看看samples中各种库都是怎么用的，比如PI_, PRM_。
#### 2.0. standalone
// 是否可以定义Hscript cmd文件来自动创建Shelf Tool
```
MOT_Director* boss = new MOT_Director("standalone");
OPsetDirector(boss);
PIcreateResourceManager();
CMD_Manager * cmd = boss->getCommandManager();
cmd->doPrompt();
cmd->sendInput("source -q 123.cmd"); // 执行123.cmd中的所有Hscript命令。
```
#### 2.1. SYS库
```
SYS_Types.h
  buf.sprintf("volume_%" SYS_PRId64, (exint)prim->getMapIndex());
SYS_Math.h
  SYSdegToGrad()
  SYScos()
SYS_Floor.h
SYS_Inline.h
SYS_FORCE_INCLINE
SYS_StaticAssert.h
SYS_TypeTraits.h
SYS_Pragma.h
```

### 2.2. UT库 Utilities
```
UT_ASSERT(xxxx);

UT_AutoInterrupt interrupt("Writing curve transforms");
if (interrupt.wasInterrupted()) {
    return;
}

UT_IntrusivePtr

```

#### 2.3. PI库 Procedural Interface 库。
PI_ResourceTemplate -> PI_StateTemplate
                    -> PI_HandleTemplate
                    -> PI_PITemplate
                    -> PI_SelectorTemplate
#### 2.4. PRM库
PRM_Template 和 PRM_Instance一起定义PRM_Parm, PRM_Template是PRM_Parm的静态成员。
PRM_Item vs PRM_Name for PRM_ChoiceList。 前者带icon。

UI_Object -> AP_Interface -> BM_SimpleState -> BM_ParmState -> BM_State -> BM_OpState -> BM_SingleOpState -> MSS_SingleOpBaseState -> MSS_SingleOpState

#### 2.5. GA库
- enums and namespaces
```
  GA_AttributeOwner values GA_ATTRIB_VERTEX, GA_ATTRIB_POINT, GA_ATTRIB_PRIMITIVE, GA_ATTRIB_GLOBAL // 
  GA_Storage : GA_STORE_REAL32 etc // in GA_Types.h
  GA_TypeInfo : GA_TYPE_HPOINT, GA_TYPE_XXX
  GA_GroupType : GA_GROUP_xxx

namespace GA_Names // in GA_Names.h
  P, Pw, N, uv, v, wCd, Alpha, xxx
```

- Attributes and Detail
```
GA_Attribute -> GA_ATINumeric //GA_ATINumeric.h
GA_ROHandleT, GA_RWHandleT //GA_Handle.h
GA_Detail // GA_Detail.h, The detail stores lists of attributes (GA_Attribute) 

GA_Primitive
  GA_Detail *myDetail;
  GA_Offset myOffset;
  GA_OffsetList myVertexList;


// Copy attributes satisfied some conditions.
//static GA_AttributeFilter	 selectStandard(const GA_Attribute *exclude=0);
GA_AttributeFilter filter = GA_AttributeFilter::selectOr(GA_AttributeFilter::selectStandard(gdp->getP()), GA_AttributeFilter::selectGroup());
GA_PointWrangler ptwrangler(*gdp, filter);
if (ptwrangler.getNumAttributes() > 0) ptwrangler.copyAttributeValues(newptoff, oldptoff);


GA_PolyCounts
GA_PointWrangler


```

#### 2.6. GEO库 is also Geometry 库
```
GA_Detail -> GEO_Detail // This defines the detail class (the container for all points/primitives)
GA_Primitive -> GEO_Primitive -> GEO_HULL
GA_PolyCounts -> GEO_PolyCounts           
```
#### 2.7. GU库： Geometry Utility 库
```
GEO_Detail -> GU_Detail // GU_Detail represents a container for geometry.  It contains primitives,
 *	points, vertices, etc. and all of the attributes. It also provides some
 *	conventient methods for manipulating geometry.

GU_Detail -> GOP_Guide
GU_DetailHandle -> GU_ConstDetailHandle
GU_DetailHandleAutoReadLock
GU_DetailHandleAutoWriteLock


GEO_PrimCircle -> GU_PrimCircle

GU_Group

GU_Curve

GU_StencilPixel
GU_BrushStencilMode
GU_BrushStencil
	UT_Array<GU_StencilPixel>	 myEntries;
	    UT_IntArray			 myStencilRef;
    	UT_IntArray			 myPointPass;
    		UT_Vector3Array		 myColours;
		UT_Array<UT_IntArray *>	*myPt2Vtx;
    	UT_Array<GA_Offset>		*myVtx;
    		const GU_Detail		*myGdp;
    	int				 myCurPixel, myCurSubIdx;
    	int				 myCurIteratePass;
    		bool			 myCurIsVertexIterate;
GU_BrushMergeMode
GU_BrushNib
GU_Brush
```

#### 2.8. GT库
 Geometry 库 使用 GA and GU的几何库。 可以理解为GT比GA和GU还要高级。
```
GDT : Geometry Data Type。
The GDT data structures provide a flexible and efficient representation of geometric primitives, attributes, and connectivity information. They are designed to handle various types of geometry, including points, curves, polygons, and volumes.

GDT包含如下这些数据结构：

/// @see GT_AttributeList
typedef UT_IntrusivePtr<GT_AttributeList>	GT_AttributeListHandle;
/// @see GT_AttributeMap
typedef UT_IntrusivePtr<GT_AttributeMap>	GT_AttributeMapHandle;
/// @see GT_DataArray
typedef UT_IntrusivePtr<GT_DataArray>		GT_DataArrayHandle;
/// @see GT_Primitive
typedef UT_IntrusivePtr<GT_Primitive>		GT_PrimitiveHandle;
/// @see GT_Transform
typedef UT_IntrusivePtr<GT_Transform>		GT_TransformHandle;
/// @see GT_TransformArray
typedef UT_IntrusivePtr<GT_TransformArray>	GT_TransformArrayHandle;
/// @see GT_FaceSetMap
typedef UT_IntrusivePtr<GT_FaceSetMap>		GT_FaceSetMapPtr;

```

#### 2.9. UI库
```

```
#### 2.10. SOP库
- enums
```
SOP_BrushEvent
    SOP_BRUSHSTROKE_BEGIN,
    SOP_BRUSHSTROKE_ACTIVE,
    SOP_BRUSHSTROKE_END,
    SOP_BRUSHSTROKE_CLICK,
    SOP_BRUSHSTROKE_NOP
SOP_BrushOp :
    SOP_BRUSHOP_UNASSIGNED,
    SOP_BRUSHOP_DEFORM,
    SOP_BRUSHOP_COMB,
    SOP_BRUSHOP_PAINT,
    SOP_BRUSHOP_SMOOTH,
    SOP_BRUSHOP_SCRIPT,
    SOP_BRUSHOP_SMOOTHDEFORM,
    SOP_BRUSHOP_EYEDROP,
    SOP_BRUSHOP_ERASE,
    SOP_BRUSHOP_SMOOTHATTRIB,
    SOP_BRUSHOP_SMOOTHNORMAL,
    SOP_BRUSHOP_CALLBACK,
    SOP_BRUSHOP_DRAGTEXTURE,
    SOP_BRUSHOP_SCALETEXTURE,
    SOP_BRUSHOP_SMOOTHTEXTURE,
    SOP_BRUSHOP_SMOOTHLAYER,
    SOP_BRUSHOP_SMOOTHSINGLE,
    SOP_BRUSHOP_REDUCE,
    SOP_BRUSHOP_ERASESINGLE,
    SOP_BRUSHOP_LIFT,
    SOP_BRUSHOP_ROTATE,
    SOP_BRUSHOP_SMUDGE,
    SOP_BRUSHOP_SCALE,
SOP_BrushShape :     SOP_BRUSHSHAPE_CIRCLE, SOP_BRUSHSHAPE_SQUARE, SOP_BRUSHSHAPE_BITMAP
    SOP_BRUSHSHAPE_CIRCLE,
    SOP_BRUSHSHAPE_SQUARE,
    SOP_BRUSHSHAPE_BITMAP
SOP_BrushVisType
    SOP_BRUSHVIS_FALSECOLOUR,
    SOP_BRUSHVIS_CAPTUREWEIGHT
```

```
SOP_Node -> SOP_GDT -> SOP_BrushBase
SOP_UndoGDT
SOP_UndoGDTOpDepend


```

#### 2.11. State


- UI_EventMethod
typedef void (UI_Object::*UI_EventMethod)(UI_Event* );

- UI_Event:
Wrapper of UI_DeviceEvent
```
UI_Value* value;
UI_Object* source;
UI_Object* target;
UI_EventMethod method;
UI_EventType type;
UI_DeviceEvent state;   // the recent event
UI_Reason reason;
UT_Array<UI_DeviceEvent> * myStateHistory; // can be 0, denotes as collapsed events

函数：
UI_Event(t, to, from, e);
UI_Event(v, to, from, r);
UI_Event(v, to, callback, from, r);
UI_Event(t, to, callback, from);

trigger()
```


- UI_Object: 所有有能力收到UI_Events的基类。 UI_Object -> UI_Value
```
有关的类型:
  UT_WorkBuffer, UT_Assert, UT_UniquePtr, UI_Event, enum UI_EventType, UI_Manager, UI_Queue, UI_DeviceEvent, UI_Value, UI_KeyDelegate, UI_HotkeyEcho,
  UI_ObjectList, UI_ValueList, UI_KeyDelegateList
  UI_EventMethod : typedef void (UI_Object::*UI_EventMethod)(UI_Event* );

成员函数：
UI_Object();

getParent();
setParent();
void addDependent()
void removeDependent();

virtual handleEvent(UI_Event* event);
void setEvent(e) cosnt;
void distributeEvent(e, int upwards);
void relayEvent(e, UI_Object* target);
void purgeEvents(t, target, method);
void triggerImmediateEvent(e, int upwards);

// interests
UI_Value::interestedInValue(UI_Value *)

/ key
UI_KeyDelegate::addKeyDelegateClientship(UI_KeyDelegate *)

// key related
static void keyEcho(...)
static toggleKeyEcho(...)

成员变量：
UI_Object* myParent
UI_ObjectList myDependents;
UI_ValueList myValueInterests;
UT_UniquePtr<UI_KeyDelegateList> myKeyDelegateClientships;
static UI_Manager *theUIManager;  // UI_Object::getManager();
static UI_Queue* theUIQueue;   // UI_Object::getInputQueue();

```

- AP_Interface: 维护named UI_Feel 和 UI_Value之间的hash表, AP means Application。
```
有关的类：
UT_Url, UT_SharedPtr, UT_SymbolMap<T>
UI_Manager, UI_Feel, UI_KeyDelegate, UI_Window

成员函数：
AP_Interface(const char* myname, const char* const * names, UI_EventMethod const* methods);
void wireInterface(UI_Manager *uims);
void unwireInterface(UI_Manager *uims);
bool readUIFile(xxx);
AP_Interface::createPreferenceFile();
void setValueSymbol(symbol, value, bool warn= true);
void setObjectSymbol(symbol, obj, bool warn= true);
UI_Feel getFeelSymbol(symbol);
void setKeyDelegateSymbol(symbol, delegate, bool warn = true);
template<typename T> T* findObject(name);
template<typename T> T* findValue(name);
virtual void initApplication(uims, argc, argv);

成员变量：
static AP_Interface::theMainApplication;
UI_NamedValueMap *myValueTable;UI_NamedObjectMap *myObjectTable;
UI_NamedKeyDelegateMap * myKeyDelegateTable;
xxx myKeyBindingProxyRequestTable;

全局函数：
SIgetObject(app, name);
SIgetValue(app, name);

```

- BM_SimpleState: handles and states， 不可实例化
```
有关的类：
RE_Render, RE_Cursor, UI_Menu, BM_SceneManager, BM_Viewport

成员函数：
BM_SimpleState(BM_SceneManager&app, cursor, name, vnames, vmethods);
getIconName
getLabel
getDescription
getHotkeyString
getStateMenu()
getSelectorMenu()
getExtraStateMenu()
onBeginRMBMenu();
enter() = 0, exit() = 0, interrupt(); resume();
virtual void handleMouseEvent(e) = 0;
virtual void handleKeyEvent(int key, e) final;
virtual void handleKeyTypeEvent(key, e, viewport);
virtual void render(r, x, y);
virtual vid overlayRender(r, x, y);
virtual int getToolboxCount() const;
virtual UI_Feel *getToolbox(index) const;
volatile state and non-volatile state
virtual bool isOverlay() const;
virtual int getVolatileToolboxCount() const;
virtual UI_Feel* getVolatileToolbox(int index) const;
// Handle ? 
virtual int isHandle() const = 0;
virtual int isModifier() const = 0;  //

sceneManager();
replaceCursor(newcursor);
unsigned int getViewportMask();

beginDistributedUndoBlock(operation, blocktype, ignore_log);
endDistributedUndoBlock(ignore_log);

void pushCursor();
void popCursor();
setDefaultCursor();
void initializeUI();

void mouseDown(e);
int mouseDown() const;

void setViewportMask(unsigned mask);
成员变量：
mySceneManager
muCursor
myPrevState
myVolatileViewport
UI_Feel *myMenuGadget;
UI_Feel * myOrthoMenuGadget;
UI_Feel *mySimpleToolbox;
myUndoWorkerFinder;
myCursorStack;
```  

- BM_ParmState: 支持parameters的state/handle, 貌似相關接口都沒有在samples中出現。 不可实例化
```
成员函数：

dialogFeel() const;
getToolboxCount();
getToolbox(int index);
virtual int disableParms();

const PRM_Parm* parameter(name) const;
PRM_ParmList *parameters() const;
UI_Value *parmValue() const;

成员变量：
myParmToolbox
myParmList
myParmVal
myToolboxParmVal
myParmDialog
myExtraParmDialog
myPresetInfo;
```

- BM_State: BM state的基类, 看isHandle()是返回0， 可实例化
```
有关的类：
DD_ChoiceList
函数：
int isHandle() const override{ return 0; }
virtual void concealState();
virtual void revealState();

virtual int preprocessSelect(e);
virtual int handleMouseEvent(e) override;
virtual int handleMouseWheelEvent(e); // 第一个处理Wheel的state
virtual int handleDoubleClickEvent(e); // 第一个处理DoubleClick的state
virtual void handleGeoChangedEvent(e);
virtual void handleDetailLookGeomChanged(BM_DetailLook *look);

virtual void render(r, x, y) override;
virtual void renderPartialOverlay(r, x, y);

virtual int generate(how, parms);
void startGenerating(how, requestnew, exsitu);
virtual void stopGenerating();

int hasOpNode(node) const;
handleOpNodeChange(node);
handleOpUIChange(node);
handleOpNetChange(network);
handleOpNetClear();

virtual int handleNodeDeleted(node);

// BM_StateFlags related functions
void wantsLocates(int yesNo);
...

virtual bool ignoreDisplayFlagChange() const;

virtual bool getAllowIndirectHandleDrag() const; // MMB can be used for indirect handle drags.

PI_StateTemplate &getTemplate() const;

virtual int isOpIndependent() const;

virtual void refreshBindings(id, op_type);

virtual bool showPersistent() const;

virtual bool getShowSelectedOp() const;

virtual UI_Feel* getCustomSelectModeSideBar() const;

virtual bool useSecureSelection() const;

virtual bool useVolatileSelection() const;

// Drag and drop support
virtual int testDragDrop(DD_Source&);
...

//
virtual bool switchHandleTool(BM_MoveTool::Type tool, bool reveal_updated_handles){return false;}

virtual void initializeHUDs();
int updateHUD(args);
void enterHUD();
void exitHUD();
void renderHUD();

成员变量：
myState;
myFlags;
static UI_Event theDelayedSelectionEvent;
PI_StateTemplate &myTemplate;
bool myCursorPushed;
int myBusyCounter;
int myUniqueId;
myQtNotifier;
myViewNotifier;
```

- BM_OpState: automated state that links handles with op parameters.


## QA
what is work bench?

what is detached attribute?

what is feel ? feel containing extra buttons of the state.

what is PIs?


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



