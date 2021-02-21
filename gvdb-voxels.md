# VolumeGVDB

## CTX
PUSH_CTX和POP_CTX

## gvdbPTX
几个cuh文件及两个cu文件。 cuh文件中包含一些类是host代码中没有的（VDBNode, VDBAtlasNode），也包含一些在host代码中存在的类(VDBInfo)。

## point cloud
```
point cloud Min, Max，p需要转成 p' = (p-Min)/(Max-Min)*65535这样的放到m_pnt1
wMin = Min
坐标需要做一次平移将使得vMin >= (0,0,0)
pntl需要将坐标转换到0~65535(ushort)
```

## Topology
即GVDB hierarchy

## Pool
group, level, and index from a pool

## Clear
干了什么? CPU上的操作。
```
// Empty VDB data (keep pools)
mPool->PoolEmptyAll ();		// does not free pool mem
mRoot = ID_UNDEFL;

// Empty atlas & atlas map
mPool->AtlasEmptyAll ();	// does not free atlas

mPnt.Set ( 0, 0, 0 );

mRebuildTopo = true;			// full rebuild required
```
## 
getRange(level) // 返回某一节的node的index-space range




## Data
```
Bounding box : mPosMin, mPosMax, mPosRange
```


## Allocator
```
Allocator *mPool
   // Pool functions
   // Texture functions
   // Atlas functions
   // Atlas Mapping
   // Neighbor Table
   // Query function

```
## Configure
<3,3,3,3,4> 4 is level 0.
## Level
## Topology
getRes(lev) //比如lev=0时，返回2^4=16
InsertPointsSubcell(subcell_size, num_pnts, pRadius, trans, pSCPntsLength)

## Node
- VDBNode
```
在cuh文件中
struct ALIGN(16) VDBNode {
	uchar		mLev;			// Level		Max = 255			1 byte
	uchar		mFlags;
	uchar		pad[2];
	int3		mPos;			// Pos			Max = +/- 4 mil (linear space/range)	12 bytes
	int3		mValue;			// Value		Max = +8 mil		4 bytes
	uint64		mParent;		// Parent ID						8 bytes
	uint64		mChildList;		// Child List						8 bytes
#ifdef USE_BITMASKS
	uint64		mMask;			// Bitmask starts
#endif
};
```
- VDBAtlasNode
```
struct ALIGN(16) VDBAtlasNode {
	int3		mPos;
	int			mLeafID;
};
```
## VDBInfo
```
1. 从VDBInfo读取数据
// Host code
void* args[4] = { &cuVDBInfo, &cellNum, &mAux[AUX_TEST_1].gpu, &mAux[AUX_TEST].gpu };
cudaCheck(cuLaunchKernel(cuFunc[FUNC_READ_GRID_VEL], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL), "VolumeGVDB", "ReadGridVel", "cuLaunch", "FUNC_READ_GRID_VEL",mbDebug);

// Kernel code
__global__ void gvdbReadGridVel (VDBInfo* gvdb, int cell_num, int3* cell_pos, float* cell_vel)

2. 向VDBInfo写数据

```
## Nodes
```
getCover: world-space covering size
getRange: index-space covering size

Vector3DI GetCoveringNode(level, pos, range/*out*/).
```
## Grp
## Level
## Subcell
## Voxels
所有叶子节点的voxels都是active的。即level 0的数据。

## Atlas vs Topology
```
- 3D Texture or CUArray
- 里面是bricks
- type and size
- 跟world space没关系
- 当空间不够的时候会，动态重新分配(resized along Z axis)。
```
## Channels
- 每个是一个atlas
## Brick
```
int bricks = static_cast<int>(mPool->getAtlas(0).usedNum);
```
## Aux
```
// 主要接口
DataPtr mAux[MAX_AUX]; // 这个东西是在GPU上的。
Allocator *mPool;
void PrepareAux ( int id, int cnt, int stride, bool bZero, bool bCPU=false );

1. AUX_PNTPOS : 通过SetPoints传递DataPtr设上去的。 看下面代码 它是 m_numpnts x sizeof(Vector3DF)
gvdb.AllocData(m_pnt1, m_numpnts, sizeof(ushort) * 3, true);
gvdb.AllocData(m_pnts, m_numpnts, sizeof(Vector3DF), true);
gvdb.CommitData(m_pnt1);
Vector3DF wdelta ( (wMax.x - wMin.x)/65535.0f, (wMax.y - wMin.y)/65535.0f, (wMax.z - wMin.z)/65535.0f );
gvdb.ConvertAndTransform ( m_pnt1, 2, m_pnts, 4, m_numpnts, wMin, wdelta, Vector3DF(0,0,0), Vector3DF(m_renderscale,m_renderscale,m_renderscale) );
gvdb.RetrieveData(m_pnts); // Copy back to the CPU so that we can locally view it
DataPtr temp;
gvdb.SetPoints( m_pnts, temp, temp);


2. AUX_BOUNDING_BOX : 6 x sizeof(float)

3. AUX_WORLD_POS_X : num_pnts x sizeof(float)

4. AUX_MARKER : numPnts x sizeof(int)
5. AUX_MARKER_PRESUM : numPnts x sizeof(int)
6. AUX_UNIQUE_CNT : 1 x sizeof(int)
7. AUX_LEVEL_CNT : pLevDepth, sizeof(int)
8. AUX_SORTED_LEVXYZ : numPnts x 4*sizeof(unsigned short)， 64bit integer
9. AUX_BRICK_LEVXYZ : numPnts x 4*sizeof(unsigned short), 64bit integer
```
## Context
## Radix Sort
## Scan
## PrefixSum
## FindUnique

