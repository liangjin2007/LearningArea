# PhysicsEngine
- [1.Physx](#1Physx)
- [2.PhysicsCore](#2PhysicsCore)
- [3.Engine.Classes.PhysicsEngine](#3Engine.Classes.PhysicsEngine)

## 1.Physx
- [physx文档](https://gameworksdocs.nvidia.com/PhysX/4.1/documentation/physxguide/Manual/Introduction.html#about-this-user-guide)
```
PhysX is designed to be robust, high performance, scalable, portable, as well as easy to integrate and use. These capabilities make PhysX suitable as a foundation technology for game engines and other real time simulation systems.

Default Cpu simulation Gpu Rigid Body example for Gpu simulation.

The basic concepts of the world within a PhysX simulation are easy to describe:
  1. The PhysX world comprises a collection of Scenes, each containing objects called Actors;
  2. Each Scene defines its own reference frame encompassing all of space and time;
  3. Actors in different Scenes do not interact with each other;
  4. Characters and vehicles are complex specialized objects made from Actors;
  5. Actors have physical state : position and orientation; velocity or momentum; energy; etc,
  6. Actor physical state may evolve over time due to applied forces, constraints such as joints or contacts, and interactions between Actors.


HelloWorld程序：SnippedHelloWorld

  #include "PxPhysicsAPI.h"
  using namespace physx;
  
  PxDefaultAllocator		gAllocator;
  PxDefaultErrorCallback	gErrorCallback;
  
  PxFoundation*			gFoundation = NULL;
  PxPhysics*				gPhysics	= NULL;
  
  PxDefaultCpuDispatcher*	gDispatcher = NULL;
  PxScene*				gScene		= NULL;
  
  PxMaterial*				gMaterial	= NULL;
  
  PxPvd*                  gPvd        = NULL;
  
  PxReal stackZ = 10.0f;


  void initPhysics(bool interactive)
  {
  	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
  
  	gPvd = PxCreatePvd(*gFoundation);
  	PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
  	gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);
  
  	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(),true,gPvd);
  
  	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
  	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
  	gDispatcher = PxDefaultCpuDispatcherCreate(2);
  	sceneDesc.cpuDispatcher	= gDispatcher;
  	sceneDesc.filterShader	= PxDefaultSimulationFilterShader;
  	gScene = gPhysics->createScene(sceneDesc);
  
  	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
  	if(pvdClient)
  	{
  		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
  		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
  		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
  	}
  	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
  
  	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
  	gScene->addActor(*groundPlane);
  
  	for(PxU32 i=0;i<5;i++)
  		createStack(PxTransform(PxVec3(0,0,stackZ-=10.0f)), 10, 2.0f);
  
  	if(!interactive)
  		createDynamic(PxTransform(PxVec3(0,40,100)), PxSphereGeometry(10), PxVec3(0,-50,-100));
  }

  void stepPhysics(bool /*interactive*/)
  {
  	gScene->simulate(1.0f/60.0f);
  	gScene->fetchResults(true);
  }

  PxRigidDynamic* createDynamic(const PxTransform& t, const PxGeometry& geometry, const PxVec3& velocity=PxVec3(0))
  {
  	PxRigidDynamic* dynamic = PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, 10.0f);
  	dynamic->setAngularDamping(0.5f);
  	dynamic->setLinearVelocity(velocity);
  	gScene->addActor(*dynamic);
  	return dynamic;
  }
  
  void createStack(const PxTransform& t, PxU32 size, PxReal halfExtent)
  {
  	PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
  	for(PxU32 i=0; i<size;i++)
  	{
  		for(PxU32 j=0;j<size-i;j++)
  		{
  			PxTransform localTm(PxVec3(PxReal(j*2) - PxReal(size-i), PxReal(i*2+1), 0) * halfExtent);
  			PxRigidDynamic* body = gPhysics->createRigidDynamic(t.transform(localTm));
  			body->attachShape(*shape);
  			PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
  			gScene->addActor(*body);
  		}
  	}
  	shape->release();
  }
  void cleanupPhysics(bool /*interactive*/)
  {
  	PX_RELEASE(gScene);
  	PX_RELEASE(gDispatcher);
  	PX_RELEASE(gPhysics);
  	if(gPvd)
  	{
  		PxPvdTransport* transport = gPvd->getTransport();
  		gPvd->release();	gPvd = NULL;
  		PX_RELEASE(transport);
  	}
  	PX_RELEASE(gFoundation);
  	
  	printf("SnippetHelloWorld done.\n");
  }

  void renderLoop()
  {
  	sCamera = new Snippets::Camera(PxVec3(50.0f, 50.0f, 50.0f), PxVec3(-0.6f,-0.2f,-0.7f));
  
  	Snippets::setupDefaultWindow("PhysX Snippet HelloWorld");
  	Snippets::setupDefaultRenderState();
  
  	glutIdleFunc(idleCallback);
  	glutDisplayFunc(renderCallback);
  	glutKeyboardFunc(keyboardCallback);
  	glutMouseFunc(mouseCallback);
  	glutMotionFunc(motionCallback);
  	motionCallback(0,0);
  
  	atexit(exitCallback);
  
  	initPhysics(true);
  	glutMainLoop();
  }

```

- 编译代码 Code Path: D:\CopyX\PhysX
  - 下载代码
  - ./generate_projects.bat生成visual studio 工程
  - 下载DXSDK_Jun10.exe安装
  - 添加DirectX SDK include和lib到Visual studio工程相关项目
  - 编译 -> 成功

### PhysX

PxPhysicsAPI.h: the main include header for the Physics SDK
- 有如下子集:
```
1 Foundation SDK/PhysXFoundation： foundation/PxXXX.h
2 Physics specific utilities and common code/PhysXCommon: common/PxXXX.h
3 Task Manager/PhysXTask: task/PxTask.h
4 CUDA Manager/PhysX: gpu/PxGpu.h
5 Geometry Library/PhysXCommon: geometry/PxXXX.h, Box, Capsule, HeightField, Plane, Sphere, Triangle, TriangleMesh, Geometry, TriangleMeshGeometry, ConvexMeshGeometry, BVH
6 Core SDK/PhysX: 无目录前缀
7 Character Controller/PhysXCharacterKinematic: characterkinematic/PxXXX.h
8 Cooking(data preprocessing)/PhysXCooking: cooking/PxXXX.h
9 Extensions to the SDK/PhysXExtensions: extensions/PxXXX.h
10 Serialization/PhysXExtensions: extensions/PxXXX.h
11 Vehicle Simulation/PhysXVehicle: vehicle/PxXXX.h
12 Converting the SDK to Visual Debugger/PhysXPvdSDK: pvd/PxXXX.h

看其中包含的头文件foundation/PsXXX.h是不对外暴露的。同理还有其他子集中的PsXXX头文件也不暴露。
```


### PhysXFoundation
```
foundation/PxSimpleTypes.h
  基础类型
    PxI64， PxU64, PxI32, PxU32, PxI16, PxU16, PxI8, PxU8, PxF32, PxF64, PxReal
    宏： PX_MAX_F32, PX_MAX_F64, PX_EPS_F32, PX_EPS_F64, PX_MAX_REAL, PX_EPS_REAL, PX_NORMALIZATION_EPSILON, PX_MAX_XXX, PX_MIN_XXX

foundation/Px.h
  前向申明

数学定义
  线性代数： PxMat44.h, PxMath.h
  几何： PxBounds3.h, PxPlane.h, 

宏/Preprocessor
  PxPreprocessor.h

平台/系统相关
  PxIntrinsics.h: 比如一些数学函数 sin, cos, sqrt, 倒数recip 1/a等

IO
  PxIO.h: PxInputStream, PxInputData, PxOutputStream

Memory
  PxMemory.h: PxMemCopy, PxMemSet, PxMemZero, PxMemMove

AllocatorCallback
  PxAllocatorCallback.h: 线程安全分配器

ErrorCallback
  PxErrorCallback.h: 允许用户重载，用来调试

Error enum
  PxErrors.h

Assert
  PxSharedAssert.h

StrideIterator
  PxStrideIterator.h

Bit Mask相关
  PxBitAndDataT<typename storageType, storageType bitMask>(storageType data, bool bit = false){ mData = bit ? storageType(data | bitMask) : data; }
  setBit() { mData |= bitMask; }
  operator storageType() { return storageType(mData &~bitmask); }
  clearBit() { mData &= ~bitMask; }
  storageType isBitSet() { return storageType(mData & bitMask); }

UnionCast
  template <class A, class B>
  PX_FORCE_INLINE A PxUnionCast(B b) 
  {
  	union AB
  	{
  		AB(B bb) : _b(bb)
  		{
  		}
  		 B _b;
  		 A _a;
  	} USE_VOLATILE_UNION u(b);
  	return u._a;
  }
```

### PhysXCommon
```
common/PxXXX.h
geometry/PxXXX.h
```



## 2.PhysicsCore

## 3.Engine.Classes.PhysicsEngine
