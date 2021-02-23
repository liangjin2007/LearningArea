# VolumeGVDB

## Node
```
// GVDB Node
// This is the primary element in a GVDB tree.
// Nodes are stores in memory pools managed by VolumeGVDB and created in the allocator class.
struct ALIGN(16) GVDB_API Node {
public:							//						Size:	Range:
	uchar		mLev;			// Tree Level			1 byte	Max = 0 to 255
	uchar		mFlags;			// Flags				1 byte	true - used, false - discard
	uchar		mPriority;		// Priority				1 byte
	uchar		pad;			//						1 byte
	Vector3DI	mPos;			// Pos in Index-space	12 byte
	Vector3DI	mValue;			// Value in Atlas		12 byte
	Vector3DF	mVRange;		// Value min, max, ave	12 byte
	uint64		mParent;		// Parent ID			8 byte	Pool0 reference
	uint64		mChildList;		// Child List			8 byte	Pool1 reference	#ifdef USE_BITMASKS
	uint64		mMask;			// Start of BITMASK.	8 byte  
								// HEADER TOTAL			64 bytes
};
```

## DataPtr
```
// Smart pointer for all CPU/GPU pointers 
struct GVDB_API DataPtr {
	DataPtr ();		
	char		type;				// data type
	char		apron;				// apron size
	char		filter;				// filter mode
	char		border;				// border mode
	uint64		max;				// max element count
	uint64		lastEle;			// total element count
	uint64		usedNum;			// used element count
	uint64		size;				// size of data
	uint64		stride;				// stride of data	
	Vector3DI	subdim;				// subdim		
	Allocator*	alloc;				// allocator instance
	char*		cpu;				// cpu pointer		
	int			glid;			// gpu opengl id
	CUgraphicsResource	grsc;			// gpu graphics resource (cuda)		
	CUarray		garray;				// gpu array (cuda)
	CUdeviceptr	gpu;				// gpu pointer (cuda)	
	CUtexObject tex_obj;				// gpu texture object
	CUsurfObject surf_obj;				// gpu surface object
};
```

## Elem
```
// Used to pack/unpack the group, level, and index from a pool reference
inline uint64 Elem ( uchar grp, uchar lev, uint64 ndx )	{ return uint64(grp) | (uint64(lev) << 8) | (uint64(ndx) << 16); }
inline uchar ElemGrp ( uint64 id )						{ return uchar(id & 0xFF); }
inline uchar ElemLev ( uint64 id )						{ return uchar((id>>8) & 0xFF); }
inline uint64 ElemNdx ( uint64 id )						{ return id >> 16; }
```

## Allocator
```
std::vector< DataPtr >				mPool[ MAX_POOL ]; // 每个vector<DataPtr>对应一个group， 每个DataPtr对应一个level。
std::vector< DataPtr >				mAtlas;
std::vector< DataPtr >				mAtlasMap;
DataPtr						mNeighbors;


bool						mbDebug;
int						mVFBO[2];
CUstream					mStream;
CUmodule					cuAllocatorModule;
CUfunction					cuFillTex;	
CUfunction					cuCopyTexC;
CUfunction					cuCopyTexF;
CUfunction					cuCopyBufToTexC;
CUfunction					cuCopyBufToTexF;
CUfunction					cuCopyTexZYX;
CUfunction					cuRetrieveTexXYZ;
```

```

```
## VolumeBase
```
Vector3DF		mObjMin, mObjMax;			// world space
Vector3DF		mVoxMin, mVoxMax, mVoxRes;		// coordinate space
Vector3DF		mVoxResMax;

Vector3DF		mRenderTime;
bool			mbProfile;
bool			mbVerbose;

DataPtr			mTransferPtr;				// Transfer function
std::vector<DataPtr>	mRenderBuf;				// Non-owning list of render buffers (since apps can add their own render buffers)

Allocator*		mPool = nullptr;			// Allocator
Scene*			mScene = nullptr;			// Scene (non-owning pointer)
```

## VolumeGVDB
```
// VDB Settings
int			mLogDim[MAXLEV];	// internal res config
Vector3DF		mClrDim[MAXLEV];
int			mApron;
Matrix4F		mXForm;
bool			mbDebug;
Vector3DI		mDefaultAxiscnt;

// Root node
uint64			mRoot;
Vector3DI		mPnt;

// Scene 
ScnInfo			mScnInfo;
CUdeviceptr		cuScnInfo;

// VDB Data Structure
VDBInfo			mVDBInfo;			
CUdeviceptr		cuVDBInfo;

bool			mHasObstacle;
CUdeviceptr		cuOBSVDBInfo; // Non-owning pointer to VDBInfo used for collision

// CUDA kernels
CUmodule		cuModule[5];
CUfunction		cuFunc[ MAX_FUNC ];

// CUDA pointers		
CUdeviceptr		cuXform;
CUdeviceptr		cuDebug;

std::vector< Vector3DF >	leaf_pos;
std::vector< uint64 >		leaf_ptr;

// Auxiliary buffers
DataPtr			mAux[MAX_AUX];		// Auxiliary
std::string		mAuxName[MAX_AUX];

Volume3D*		mV3D;			// Volume 3D

// Dummy frame buffer
int mDummyFrameBuffer;

// CUDA Device & Context
int			mDevSelect;
CUcontext		mContext;
CUdevice		mDevice;
CUstream		mStream;

bool			mRebuildTopo;
int			mCurrDepth;
Vector3DF		mPosMin, mPosMax, mPosRange;
Vector3DF		mVelMin, mVelMax, mVelRange;

Vector3DI		mSimBounds;
float			mEpsilon;
int			mMaxIter;

// Grid Transform
Vector3DF		mPretrans, mAngs, mTrans, mScale;
Matrix4F		mXform, mInvXform, mInvXrot;

const char*		mRendName[SHADE_MAX];

float			m_bias;
```

## 概念
- 每个level有两个group， group 1 存的是childlist


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
## Range
getRange(level) // 返回某一节的node的index-space range

## Node
```
inline __device__ int3 GetCoveringNode (float3 pos, int3 range)
{
	int3 nodepos;

	nodepos.x = ceil(pos.x / range.x) * range.x;
	nodepos.y = ceil(pos.y / range.y) * range.y;
	nodepos.z = ceil(pos.z / range.z) * range.z;
	if ( pos.x < nodepos.x ) nodepos.x -= range.x;
	if ( pos.y < nodepos.y ) nodepos.y -= range.y;
	if ( pos.z < nodepos.z ) nodepos.z -= range.z;

	return nodepos;
}
```
```
```
## Data
```
Bounding box : mPosMin, mPosMax, mPosRange  这些是Index-space的范围。
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
8. AUX_SORTED_LEVXYZ : numPnts x pRootLev x 4*sizeof(unsigned short)， 64bit integer
9. AUX_BRICK_LEVXYZ : numPnts x pRootLev x 4*sizeof(unsigned short), 64bit integer
10. AUX_RANGE_RES : pRootLev x sizeof(int)
11. 
```
## Context
## Radix Sort
## Scan
## PrefixSum
给定一个数组A[1..n]，前缀和数组PrefixSum[1..n]定义为：PrefixSum[i] = A[0]+A[1]+...+A[i-1]；
例如：A[5,6,7,8] --> PrefixSum[5,11,18,26]

## FindUnique

