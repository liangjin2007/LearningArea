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

## 看HDK文档看看HDK能做什么
### DM库

#### Custom Viewport Rendering https://www.sidefx.com/docs/hdk/_h_d_k__viewport.html
```
DM_RenderTable
DM_PrimitiveHook
DM_SceneHook
DM_SceneRenderHook
void
newRenderHook(DM_RenderTable *table)
{
    // Register this hook as a replacement for the background rendering. 
    table->registerSceneHook(new DM_BackgroundHook,
			     DM_HOOK_BACKGROUND,
			     DM_HOOK_REPLACE_NATIVE);

    // Note that the name or label doesn't have to match the hook's name.
    table->installSceneOption("checker_bg", "Checkered Background");
}


```
#### Custom Viewport Event Handling
DM_EventTable
DM_MouseHook


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

GU_PrimPoly vs GU_PrimMesh
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

- State
```
UI_Object -> AP_Interface -> BM_SimpleState -> BM_ParmState -> BM_State -> BM_OpState - > BM_SingleOpState -> MSS_SingleOpBaseState-> MSS_SingleOpState -> MSS_BrushBaseState
AP_Interface
    UI_NamedValueMap	 *myValueTable;
    UI_NamedObjectMap	 *myObjectTable;
    UI_NamedKeyDelegateMap	 *myKeyDelegateTable;
    NamedProxyRequestMap	 *myKeyBindingProxyRequestTable;	
BM_SimpleState
    BM_SceneManager 	&mySceneManager;// The sceneManager I'm a state of.
    const char		*myCursor;	// the name of our cursor.
    BM_SimpleState	*myPrevState;	// The state we're preempting
    BM_Viewport		*myVolatileViewport; // the viewport that events
					     // dealing with menus and other
					     // volatile handles should go to
    UI_Feel		*myMenuGadget;	   // the feel that holds the menu
    UI_Feel		*myOrthoMenuGadget;// the feel that holds the menu to
					   // be used in ortho viewports
    int			myDistributedUndoBlockLevel;
    UI_Feel		*mySimpleToolbox;
    unsigned int	 myViewportMask; // bit mask for viewports in which
					 // we need to handle events.
    int			 myMouseDown; // mouse button down on START or PICKED
    UT_UndoWorkerFinder<BM_SimpleState> myUndoWorkerFinder;
    UT_Array<RE_Cursor*> myCursorStack;
BM_ParmState
    UT_String		 myName; 	// name 
    UT_String		 myEnglishName; // name in English
    PSI2_DialogPRMExported *myParmToolbox;
    PRM_ParmList	*myParmList;	// the parms inside the dialog
    UI_Value		*myParmVal;	// value to communicate with parms
    UI_Value		*myToolboxParmVal;
    PSI2_DialogPRM	*myParmDialog;	// the dialog that holds the parms
    PSI2_DialogPRM	*myExtraParmDialog; // an extra copy of that dialog.
    PRM_PresetInfo	*myPresetInfo;
    unsigned		 myOwnParmsFlag:1,  // own its parm list and dialog?
			 mySaveParmForUndoFlag:1; //should save parm for undo?
BM_State
    PI_StateTemplate	&myTemplate;
    bool		myCursorPushed;
    int			myBusyCounter;
    int			myUniqueId;
    // HUD info updates queued while this state was being constructed in
    // BM_ResourceManager::newState().
    UT_Array<HUDInfoArgsCopyUPtr>	myNewStateHUDQueue;
    // Specific data member for handling the HUD notifications.
    UT_SharedPtr<bmQtNotifier>	myQtNotifier;
    UT_SharedPtr<bmViewNotifier> myViewNotifier;

BM_OpState : virtual class
    BM_OpView				&myViewer;
    UT_ValArray<opbm_DialogInfo *>	 myDialogs;
    UT_ValArray<opbm_PIContext *>	 myPIs;
    UT_ValArray<UI_Feel *>		 myMiscFeels;
    SI_Folders				*myFolders;
    static const char			*STATE_DIALOG_FOLDER;
    static const char			*HANDLES_FOLDER;
    static const char			*OP_DIALOG_FOLDER;
    static int				 theAutoHighlightFlag;
    UT_Map<std::pair<int32 /*folder id*/, int64 /*ui cache id*/>, opbm_UIInfo> myUIInfoMap;

BM_SingleOpState； virtual class
   // data
    int			 myOpNodeId;
    OP_Node		*mySavedOpNode;
    OPUI_Dialog		*myOpToolbox;
    UT_SymbolMap<int>	 myHandleTable;

    // As long as this state is alive, remember what PIs are visible
    UT_BitArray		 myHandleVisibility;

    UT_String		 myRestartInfoFile;
    UT_IntArray		 myRestartOpInputs;
    UT_StringArray	 myRestartOpIndirectInputs;
    int			 myRestartOpId;

	
PI_BindingSelectorInfo
    UT_String		 myName;
    UT_String		 myDescription;
    UT_String		 myPrompt;
    UT_String		 myOpParm;
    UT_String		 myMenu;
    UT_String		 myPrimMask;
    int			 myOpInput;
    bool		 myOpInputReq;
    bool		 myAllowDrag;
    bool		 myAstSelAll;
    UT_String		 myExtraInfo;

MSS_SelectorBind
	OP3D_InputSelector		*mySelector;
	const PI_BindingSelectorInfo	*mySelectorInfo;

MSS_SingleOpState
    // info about selectors
    UT_Array<MSS_SelectorBind>		 mySelectors;
    OP3D_InputSelector			*myCurrentSelector;
    const PI_BindingSelectorInfo	*myCurrentSelectorInfo;
    // Selector used for selector hotkeys when in quickselect mode and 
    // secure selection is turned off.
    OP3D_InputSelector			*myHotkeySelector;
    UI_Menu *				 myHotkeySelectorMenu;
    // The selector index is the index of the current selector in the list of
    // selectors provided for this state.  It coincides with the index into
    // the gdp's set of temporary selections.
    int					 mySelectorIndex;
    UT_ValArray<UT_StringArray *>	 myReselectPathLists;
    UT_StringArray			 myReselectSelectionStrings;
    UT_Array<GA_GroupType>	 	 myReselectSelectionTypes;
    UI_Value				 mySelFinishedValue;
    float				 mySelectorActiveCoords[2];
    UI_Event				 myRapidFireActiveEvent;
    UI_Value				 myRapidFireActiveEventValue;
    mss_InputSelectorUndoWorker *	 mySelectorUndoWorker;
    int					 mySelectableFlag; // uses selectors?
    unsigned				 myMouseTakenFlag:1; // can rapid-fire?
    bool				 myFirstSelectionFlag;
    bool				 myInNonSecureUndo;
    bool				 myInNonSecureSelector;
    bool				 myAllowExportingCookSelectionType;
    bool				 myHasGeoChangedInterest;
    int					 myDoubleClickUndoLevel;

MSS_SingleOpBaseState
	nothing

MSS_BrushBaseState
    DM_ModifierKeys	 myModifierKeys;
    DM_ModifierKeys	 myFinishModifierKeys;
    DM_ModifierKeys	 myWheelModifierKeys;
    SOP_BrushBase	*mySavedBrushNode;
    GU_RayIntersect 	 myRayIntersect;
    DM_Detail		 myBrushHandle;
    GU_Detail		 myBrushCursor;
    UT_Matrix4		 myBrushCursorXform;
    bool		 myRenderBrushCursor;
    bool		 myBrushCursorIsUV;
    bool		 myOneHit;
    bool		 myLocatedFlag;
    UT_Vector2		 myOldCoord;

    // These are used track the resizing of the cursor in the viewport
    bool		 myResizingCursor;
    int			 myLastCursorX, myLastCursorY;

    // This is used to stash the last valid cursor orientation.
    // This allows us to rebuild the orientation to resize on
    // the users request without reintersecting.
    bool		 myOldOrientValid;
    fpreal		 myOldOrientT;
    UT_Vector3		 myOldOrientHitPos;
    UT_Vector3		 myOldOrientHitNml;
    float		 myOldOrientScaleHistory;
    bool		 myOldOrientIsUV;
    GA_Index		 myOldOrientHitPrim;
    float		 myOldOrientHitU;
    float		 myOldOrientHitV;
    float		 myOldOrientHitW;

    UI_Value		 myPrimaryButtonVal;
    UI_Value		 mySecondaryButtonVal;
    UI_Value		 myBrushShapeVal;
    UI_Value		 myBrushOrientVal;
    UI_Value		 myAccumStencilVal;

```

- Selector
```
UI_Object -> AP_Interface -> BM_InputSelector -> DM_InputSelector -> OP3D_InputSelectorBase -> OP3D_InputSelector -> OP3D_GenericSelector

BM_InputSelector
    PI_SelectorTemplate	&myTemplate;
    BM_View		*myBaseViewer;
    const char		*myBumpedCursor; // cursor I'm replacing when active

DM_InputSelector: virtual class
OP3D_InputSelectorBase
    Proxy			*myProxy;
    DM_Workbench		*myWorkbench;

    UI_Value			*myFinishedValue;
    UI_Value			*myLocatedValue;
    UI_Value			*mySelectionStyle;
    UI_Value			*myVisiblePickValue;
    UI_Value			*myContainedPickValue;
    UI_Value			*mySelectionRule;

    SI_RubberBox		*myPickBox;	// for box selection
    SI_Lasso			*myPickLasso;	// for lasso selection
    SI_Brush			*myPickBrush;	// for brush selection
    SI_Brush			*myPickLaser;	// for laser selection

    DM_SelectMode		 myPreferredSelectMode;
    bool			 myAllowDragging;
    bool			 myAllowFinishingFlag;
    bool			 myAllowFinishWithExistingSelection;
    bool			 myAllowQuickSelect;
    bool			 myAllowEmptyQuickSelect;
    bool			 myJustDisplayedOpFlag;
    bool			 myActivePicking;

    UT_StringMap<UT_IntArray>	 myPriorObjSelections;
    bool			 myCreatedPriorSelections;
 
    static bool			 thePickingMenuOn;
    static bool			 theAllowUseExistingSelection;
    static bool			 theSelectFullLoops;

    // Drawable selection
    bool			 myDrawableSelectableFlag;
    UT_StringArray		 myDrawableMask;

OP3D_InputSelector
   // Hotkey methods
    static UI_HotkeyHelper::Entry   theHotkeyList[];

    UI_HotkeyHelper	 myHotkeyHelper;
    DM_Viewport		*myHotkeyViewport;

    UT_String		 myCurrentPrompt;
    UT_String		 myDefaultPrompt;

    UT_String		 myCreatorStateName;

    UI_Value		*myGeoChangedValue;
    UI_Value		*mySelectionTypeValue;	// prims, points, etc.
    UI_Value		*myFullSelection;	// select whole gdp
    UI_Value		*myAlwaysLocate;	// always do locating

    // A selector can optionally be "sloppy" as described in the comment for
    // setSloppyPick(), whereby the user can pick any of the component types
    // allowed by mySloppyPickMask (automatically built from myAllowedTypes).
    // Once a component is picked in this mode, mySloppySelectionType will be
    // set and only components of that type can be selected until selections
    // are cleared.
    unsigned		 mySloppyPickMask;
    GA_GroupType	 mySloppySelectionType;
    GA_GroupType	 mySloppyFallbackSelectionType;
    bool		 mySloppyPick;
    bool		 mySloppySelectionTypeIsSet;

    // When overriding the values indicated by the UI buttons for the
    // above, keep previous values so we can restore.
    bool		 myCustomSelValFlag;
    int			 mySavedSelType;
    int			 mySavedSelRule;
    int			 mySavedSelStyle;
    int			 mySavedFullSel;
    int			 mySavedAlwaysLocate;

    GEO_PrimTypeCompat::TypeMask myPrimMask;	// polygon, nurbs, etc.
    
    // NB: The relative order of the selection infos only matters when the
    //     individual selections have the same pick order set.
    UT_Array<OP3D_SelectionInfo> mySelectionInfos;
    UT_Map<InfoKey, int>	 mySelectionInfoLookup;

    UT_IntArray			 mySelectedInfoIndices;
    int				 myNextPickOrder;

    typedef OP3D_SelectionManager::ComponentScopeKey ScopeKey;
    ScopeKey			 myScope;
    
    int			 myLastMouseDown;	// needed for changed events
    int			 myLastMouseStartX;	//   "     "     "      "
    int			 myLastMouseStartY;	//   "     "     "      "

    bool                 myResizingCursor;      // Cursor resize drag active.

    int			 myNodeIdForReselecting;// reselecting for this node

    bool		 myUseExistingTempSelection;
    bool		 myUseExistingCookSelection;
    bool		 myStashSelectionOnFinish;
    bool		 myInputRequiredFlag;	// is an input op required?
    bool		 myAllowDragSelFlag;	// allow box/lasso selecting?
    bool		 myFullSelFlag;		// do only full selections?
    bool		 mySaveUndosFlag;	// save undo information?
    bool		 myUseAsteriskToSelectAll; // use '*' to select all?
    bool		 myUsePrimsInEdgeSelectionFlag; // use primitives when
						// selecting edges (e.g. 0e1)
    bool		 myPickAtObjLevelFlag;	// pick geo at OBJ level
    bool		 myAllowEdgeRingSelection;
    int		 	 myOffsetVertexMarkersOverride;
    int		 	 myOffsetVertexMarkersSaved;

    // Flag to track whether the auto converted selections stored in the
    // selection info have been set.
    bool		 myAutoConvertedSelectionsFlag;

    // After we finish selecting we must remember our type.
    PI_GeometryType	 myFinishGeometryType;
    int			 myFinishGroupTypeMenuVal;

    // Component type of current selection.
    PI_GeometryType	 myCurrentComponentType;

    // A flag to track whether this selector is currently updating the geometry
    // type buttons in setGeometryType().
    bool		 myUpdatingGeometryTypeButtons;

    bool		 myHadDoubleClick;
    
    struct InitialSelection
    {
	GA_GroupType type;
	int index;
	UT_StringHolder selection_string;
    };
    UT_StringMap<InitialSelection>	 myInitialSelections;

    OP3D_InputSelectorUndoWorker	*myUndoWorker;
    bool				 myOwnUndoWorker;
    PI_GeometryTypeArray		 myAllowedTypes;

    HeldHotkeyCacheUPtr			 myHeldHotkeyCache;

    // Utility for edge loops.  The loop start pick is persistent across
    // multiple locate events, and so myLoopStartPickPath should be used
    // to identify the geometry to use with myLoopStartPick instead of
    // myLoopStartPick.getLookId() and myLoopStartPick.getDetailIndex().
    // To help avoid unnecessary lookups using the path, we track when
    // we've already updated the myLoopStartPick record to match the path
    // across extended operations in myLoopStartPickRecordMatchesPath.
    OP3D_EdgeLoopHelper			*myEdgeLoopHelper;
    UT_String				 myLoopStartPickPath;
    GR_PickRecord			 myLoopStartPick;
    GR_PickRecord			 myLoopPrevPick;
    bool				 myLoopStartPickOnlyLocated;
    bool				 myLoopStartPickRecordMatchesPath;
    OP3D_ValidForPickFilter		 myValidForPickFilter;
    void				*myValidForPickFilterData;

    LocateFilter	 myLocateFilter 			= nullptr;
    bool		 myAllowMultiPickLoopStart		= false;

    // A map from SOP node ID and detail handle index to a helper class for
    // pattern selections.
    UT_Map<std::pair<int, int>, OP3D_PatternSelectHelper*> myPatternHelpers;

    // Drawable selection
    GUI_DetailLookPtr myDrawablePicker;

OP3D_GenericSelector
    no member
```

#### 2.10. SOP库
```
```
#### 2.11. BM库和MSS库


#### 2.12. 刷子相关
- [Brush tools documentation](https://www.sidefx.com/docs/houdini/basics/brush.html)

- Geometry creation: SOP_Star.C
```  
auto &&sopparms = cookparms.parms<SOP_StarParms>();
GU_Detail *detail = cookparms.gdh().gdpNC();

// We need two points per division
exint npoints = sopparms.getDivs()*2;

if (npoints < 4)
{
// With the range restriction we have on the divisions, this
// is actually impossible, (except via integer overflow),
// but it shows how to add an error message or warning to the SOP.
cookparms.sopAddWarning(SOP_MESSAGE, "There must be at least 2 divisions; defaulting to 2.");
npoints = 4;
}

// If this SOP has cooked before and it wasn't evicted from the cache,
// its output detail will contain the geometry from the last cook.
// If it hasn't cooked, or if it was evicted from the cache,
// the output detail will be empty.
// This knowledge can save us some effort, e.g. if the number of points on
// this cook is the same as on the last cook, we can just move the points,
// (i.e. modifying P), which can also save some effort for the viewport.

GA_Offset start_ptoff;
if (detail->getNumPoints() != npoints)
{
// Either the SOP hasn't cooked, the detail was evicted from
// the cache, or the number of points changed since the last cook.

// This destroys everything except the empty P and topology attributes.
detail->clearAndDestroy();

// Build 1 closed polygon (as opposed to a curve),
// namely that has its closed flag set to true,
// and the right number of vertices, as a contiguous block
// of vertex offsets.
GA_Offset start_vtxoff;
detail->appendPrimitivesAndVertices(GA_PRIMPOLY, 1, npoints, start_vtxoff, true);

// Create the right number of points, as a contiguous block
// of point offsets.
start_ptoff = detail->appendPointBlock(npoints);

// Wire the vertices to the points.
for (exint i = 0; i < npoints; ++i)
{
    detail->setVertexPoint(start_vtxoff+i,start_ptoff+i);
}

// We added points, vertices, and primitives,
// so this will bump all topology attribute data IDs,
// P's data ID, and the primitive list data ID.
detail->bumpDataIdsForAddOrRemove(true, true, true);
}
else
{
// Same number of points as last cook, and we know that last time,
// we created a contiguous block of point offsets, so just get the
// first one.
start_ptoff = detail->pointOffset(GA_Index(0));

// We'll only be modifying P, so we only need to bump P's data ID.
detail->getP()->bumpDataId();
}

// Everything after this is just to figure out what to write to P and write it.

const SOP_StarParms::Orient plane = sopparms.getOrient();
const bool allow_negative_radius = sopparms.getNradius();

UT_Vector3 center = sopparms.getT();

int xcoord, ycoord, zcoord;
switch (plane)
{
case SOP_StarParms::Orient::XY:         // XY Plane
    xcoord = 0;
    ycoord = 1;
    zcoord = 2;
    break;
case SOP_StarParms::Orient::YZ:         // YZ Plane
    xcoord = 1;
    ycoord = 2;
    zcoord = 0;
    break;
case SOP_StarParms::Orient::ZX:         // XZ Plane
    xcoord = 0;
    ycoord = 2;
    zcoord = 1;
    break;
}

// Start the interrupt scope
UT_AutoInterrupt boss("Building Star");
if (boss.wasInterrupted())
return;

float tinc = M_PI*2 / (float)npoints;
float outer_radius = sopparms.getRad().x();
float inner_radius = sopparms.getRad().y();

// Now, set all the points of the polygon
for (exint i = 0; i < npoints; i++)
{
// Check to see if the user has interrupted us...
if (boss.wasInterrupted())
    break;

float angle = (float)i * tinc;
bool odd = (i & 1);
float rad = odd ? inner_radius : outer_radius;
if (!allow_negative_radius && rad < 0)
    rad = 0;

UT_Vector3 pos(SYScos(angle)*rad, SYSsin(angle)*rad, 0);
// Put the circle in the correct plane.
pos = UT_Vector3(pos(xcoord), pos(ycoord), pos(zcoord));
// Move the circle to be centred at the correct position.
pos += center;

// Since we created a contiguous block of point offsets,
// we can just add i to start_ptoff to find this point offset.
GA_Offset ptoff = start_ptoff + i;
detail->setPos3(ptoff, pos);
}

```

- Brush Related Classes
```
GA_GBElement -> GA_GBPoint -> GEO_Point
	    const GA_IndexMap	*myIndexMap;
    	    GA_Offset		 myOffset;


GA_Detail -> GEO_Detail -> GU_Detail
	这些类的实际数据都在基类GA_Detail中。
GA_Primitive -> GEO_Primitive -> GEO_TriMesh -> GEO_Face -> GEO_PrimPoly
					     -> GEO_PrimTriStrip
					     -> GEO_PrimTriFan
					     -> GEO_PrimPolySoup
                              -> GEO_Hull -> GEO_PrimMesh
                              -> GEO_PrimitiveVolume
	                      -> GEO_Quadric -> GEO_PrimSphere
                              -> GEO_PrimVDB
				-> GEO_VolumeElementBase -> GEO_PrimTetrahedron
			      -> GEO_PrimTriBezier
	这些类的实际数据都在基类GA_Primitive中。
		GA_Detail *myDetail;
	    	GA_Offset myOffset;
	    	GA_OffsetList myVertexList;


GU_RayIntersect
	需要熟悉这个的用法


GU_StencilPixel
GU_BrushStencilMode: enum
GU_BrushStencil： D->GU_StencilPixel
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
GU_BrushMergeMode: enum
GU_BrushNib
GU_Brush: myBrush
   const GA_PrimitiveGroup	*myGroup;
    const GA_PointGroup	*myPointGroup;
    GU_Detail		*myGdp;
    const GU_Detail	*myIsectGdp;
    bool		 myPtNmlSet;
    UT_Vector3Array	 myPtNmlCache;
    GEO_PointTree	*myPointTree;
    GEO_PointTree	*myUVPointTree;
    UT_StringHolder      myUVTreeAttribName;
    UT_Vector3Array	 myPointPos;
    UT_Vector3Array	 myUVPointPos;
    UT_Array<UT_Array<GA_Index> > myFloodConnected;
    // Vertex offsets into myIsectGdp
    UT_Array<GA_Offset>				myIsectVtx;
    // Vertex offsets into myGdp
    UT_Array<GA_Offset>				myGeoVtx;
    UT_Array<UT_IntArray *>			myPt2Vtx;
    // This assigns each vertex to a unique number depending on
    // its texture coordinate & point number.  Thus, vertices with
    // matching texture coordinates & point numbers will have
    // the same class.
    UT_IntArray					myVtxClass;
    // First ring of each point...
    UT_Array<GA_OffsetArray>			myRingZero;
    // This is the number of edges attached to each point ring.
    // If it is less than twice the myRingZero.entries(), we have
    // a boundary point.
    UT_IntArray					myRingValence;
    UT_Array<UT_Array<GA_Offset> *>		myVtxRingZero;
    UT_Array<UT_IntArray *>			myVtxRingClass;
    GA_RWHandleF	 myColourAttrib;
    GA_RWHandleV3	 myColourAttribV3;
    int			 myColourSize;
    bool		 myColourVertex;
    GA_RWHandleF	 myAlphaAttrib;
    bool		 myAlphaVertex;
    GA_RWHandleV3 	 myTextureAttrib;
    bool		 myTextureVertex;
    GA_RWHandleV3	 myNormalAttrib;
    GA_RWHandleV3	 myVisualizeAttrib;
    float		 myVisLow, myVisHigh;
    UT_ColorRamp	 myVisMode;
    bool		 myWriteAlpha;
    bool		 myUseCaptureRegion;
    int			 myCaptureIdx;
    bool                 myNormalizeWeight;
    GEO_Detail::CaptureType myCaptureType;
    bool		 myUseVisibility;
    GU_BrushMergeModeCallback	 myMergeModeCallback;
    void			*myMergeModeCallbackData;
    GU_BrushStencil			myStencil;


SOP_BrushEvent
SOP_BrushOp :
SOP_BrushShape :     SOP_BRUSHSHAPE_CIRCLE, SOP_BRUSHSHAPE_SQUARE, SOP_BRUSHSHAPE_BITMAP
SOP_BrushVisType

继承关系： SOP_Node -> SOP_GDT -> SOP_BrushBase
SOP_GDT
    // Current and Cumulative ("Permanent") deltas;
    // the permanent one is saved to disk.
    GDT_Detail		*myPermanentDelta;
    GDT_Detail		*myCurrentDelta;
    bool		 myCookedFlag; // have we cooked yet?
    bool		 myNotifyCacheFlag;
    // Selection group
    const GA_Group	*myGroup;
    // myNewOpLabel cannot be static or else we will have only one copy for
    // all derived classes!
    UT_String		 myNewOpLabel; // label for undos


SOP_BrushBase
    UT_Vector3		 myLastPos;
    UT_Vector3		 myBrushDir;
    // This stores the last uv location chosen in the 3d viewport.
    GA_Index		 myPendingLastUVPrimitive;
    UT_Vector3		 myPendingLastUVPrimitiveUV;
    // This is the last uv location chosen in the 3d viewport that
    // was cooked with.
    GA_Index		 myLastUVPrimitive;
    UT_Vector3		 myLastUVPrimitiveUV;
    // This is the last uv position that was cooked with.
    bool		 myLastUVPosValid;
    UT_Vector3		 myLastUVPos;
    UT_Vector3		 myUVBrushDir;
    UT_Vector3		 myUVBrushPos;
    GU_Brush	 	 myBrush;
    GU_RayIntersect	*myRayIntersect;
    TIL_TextureMap	*myNibFile;
    GU_Detail		*myBrushCursor;
    UT_Matrix3		 myCursorOrient;
    // These two are intentionally ephemeral variable which is NOT saved.
    // The state tweaks this to turn on/off visualization.
    bool	 	 myForceVisualize;
    int			 myIsectNodeId;
    /// These track our last geo and last isectgeo to see if it has changed.
    /// @{
    int                  myLastGDPId;
    int                  myLastIsectGDPId;
    GA_Size              myLastIsectGDPPtCount;
    int64                myLastIsectGDPTopoId;
    int64                myLastIsectGDPPrimId;
    int64                myLastIsectGDPPId;
    int64                myLastIsectGDPVisId;
    int64                myLastIsectGDPPrGrpId;
    int64                myLastIsectGDPPtGrpId;
    /// @}
    // This is so our callbacks know our cook time...
    fpreal		 myCookTime;
    bool		 myHitInUV;
    UT_StringHolder      myUVAttribName;
    GDT_Detail		 myMirroredDelta;
    GDT_MirrorTransform	 myMirrorTransform;
    SOP_BrushOp		 myCachedBrushOp[SOP_BRUSH_NUM_PENS];


SOP_UndoGDT
SOP_UndoGDTOpDepend
```




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



