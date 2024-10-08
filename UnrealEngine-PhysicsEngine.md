# PhysicsEngine
- [1.Physx](#1Physx)
- [2.PhysicsCore](#2PhysicsCore)
- [3.Engine.Classes.PhysicsEngine](#3Engine.Classes.PhysicsEngine)

## 1.Physx
- 名字空间namespace physx

- PhysXFoundation
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

## 2.PhysicsCore

## 3.Engine.Classes.PhysicsEngine
