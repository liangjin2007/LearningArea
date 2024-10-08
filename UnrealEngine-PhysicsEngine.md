# PhysicsEngine
- [1.Physx](#1Physx)
- [2.PhysicsCore](#2PhysicsCore)
- [3.Engine.Classes.PhysicsEngine](#3Engine.Classes.PhysicsEngine)

## 1.Physx
- 名字空间namespace physx

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
