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
## VDBInfo
```
struct ALIGN(16) VDBInfo {
	int			dim[MAXLEV]; // Log base 2 of lateral resolution of each node per level
	int			res[MAXLEV]; // Lateral resolution of each node per level
	Vector3DF		vdel[MAXLEV]; // How many voxels on a side a child of each level covers
	Vector3DI		noderange[MAXLEV]; // How many voxels on a side a node of each level covers
	int			nodecnt[MAXLEV]; // Total number of allocated nodes per level
	int			nodewid[MAXLEV]; // Size of a node at each level in bytes
	int			childwid[MAXLEV]; // Size of the child list per node at each level in bytes
	CUdeviceptr		nodelist[MAXLEV]; // GPU pointer to each level's pool group 0 (nodes)
	CUdeviceptr		childlist[MAXLEV]; // GPU pointer to each level's pool group 1 (child lists)
	CUdeviceptr		atlas_map; // GPU pointer to the atlas map (which maps from atlas to world space)
	Vector3DI		atlas_cnt; // Number of bricks on each axis of the atlas
	Vector3DI		atlas_res; // Total resolution in voxels of the atlas
	int			atlas_apron; // Apron size
	int			brick_res; // Resolution of a single brick
	int			apron_table[8]; // Unused
	int			top_lev; // Top level (i.e. tree spans from voxels to level 0 to level top_lev)
	int			max_iter; // Unused
	float			epsilon; // Epsilon used for voxel ray tracing
	bool			update; // Whether this information needs to be updated from the latest volume data
	uchar			clr_chan; // Index of the color channel for rendering color information
	Vector3DF		bmin; // Inclusive minimum of axis-aligned bounding box in voxels
	Vector3DF		bmax; // Inclusive maximum of axis-aligned bounding box in voxels
	CUtexObject		volIn[MAX_CHANNEL]; // Texture reference (read plus interpolation) to atlas per channel
	CUsurfObject	volOut[MAX_CHANNEL]; // Surface reference (read and write) to atlas per channel
};
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
- levels
```
int levs = mPool->getNumLevels ();
```

- nodes
```
mPool每个level有两个group， group 0 是node数据 group 1 存的是childlist
mPool的group 0为node数据

Topology有level的概念
Atlas有通道的概念，对应于level=0的情况。

// nodes
for (int n=0; n < levs; n++ ) {		
	// node counts
	numnodes_at_lev = mPool->getPoolUsedCnt(0, n);// 当n = 0时, 等于active brick count
	maxnodes_at_lev = mPool->getPoolTotalCnt(0, n);			
	numnodes_total += numnodes_at_lev;
	maxnodes_total += maxnodes_at_lev;
	gprintf ( "   Level %d: %8d %8d  %8d MB\n", n, numnodes_at_lev, maxnodes_at_lev );		
}

float MB = 1024.0*1024.0;	// convert to MB
gprintf ( "   Percent Pool Used: %4.2f%%%%\n", float(numnodes_total)*100.0f / maxnodes_total );	
```

- atlas
```
// atlas, bricks, aprons
// getAtlas(chan)通道
leafdim = static_cast<int>(mPool->getAtlas(0).stride);		// voxel res of one brick
Vector3DI axiscnt = mPool->getAtlas(0).subdim;			// number of bricks in atlas
Vector3DI axisres = axiscnt * (leafdim + mApron*2);		// number of voxels in atlas
gprintf ( "   Atlas Res:     %d x %d x %d  LeafCnt: %d x %d x %d  LeafDim: %d^3\n", axisres.x, axisres.y, axisres.z, axiscnt.x, axiscnt.y, axiscnt.z, leafdim );
int sbrk = axiscnt.x*axiscnt.y*axiscnt.z;			// number of bricks stored in atlas

Vector3DI vb = mVoxResMax;
vb /= Vector3DI(leafdim, leafdim, leafdim);
long vbrk = vb.x*vb.y*vb.z;					// number of bricks covering bounded world domain

uint64 abrk = mPool->getPoolUsedCnt(0, 0); // number

gprintf ( "   Vol Extents:   %d bricks,  %5.2f million voxels\n", vbrk, float(vbrk)*leafdim*leafdim*leafdim / 1000000.0f );
gprintf ( "   Atlas Storage: %d bricks,  %5.2f million voxels\n", sbrk, float(sbrk)*leafdim*leafdim*leafdim / 1000000.0f );
gprintf ( "   Atlas Active:  %d bricks,  %5.2f million voxels\n", abrk, float(abrk)*leafdim*leafdim*leafdim / 1000000.0f );
gprintf ( "   Occupancy:     %6.2f%%%% \n", float(abrk)*100.0f / vbrk );
```

- atlas map
```
void Allocator::AllocateAtlasMap ( int stride, Vector3DI axiscnt )
{
	DataPtr q; 
	if ( mAtlasMap.size()== 0 ) {
		q.cpu = 0; q.gpu = 0; q.max = 0;
		mAtlasMap.push_back( q );
	}
	q = mAtlasMap[0];
	if ( axiscnt.x*axiscnt.y*axiscnt.z == q.max ) return;	// same size, return

	// Reallocate atlas mapping 	
	q.max = axiscnt.x * axiscnt.y * axiscnt.z;	// max leaves supported
	q.subdim = axiscnt;
	q.usedNum = q.max;
	q.lastEle = q.max;
	q.stride = stride;
	q.size = stride * q.max;					// list of mapping structs			
	if ( q.cpu != 0x0 ) free ( q.cpu );
	q.cpu = (char*) malloc ( q.size );				// cpu allocate		
			
	size_t sz = q.size;							// gpu allocate
	if ( q.gpu != 0x0 ) cudaCheck ( cuMemFree ( q.gpu ), "Allocator", "AllocateAtlasMap", "cuMemFree", "", mbDebug);
	cudaCheck ( cuMemAlloc ( &q.gpu, q.size ), "Allocator", "AllocateAtlasMap", "cuMemAlloc", "", mbDebug );

	mAtlasMap[0] = q;
}
```

- 

- subdim是什么时候修改的？
```

```
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
```
getLD(level)    // return mLogDim
getRes(level)   // return 1 << mLogDim
getRange(level) // 返回某一节的node的index-space range
getVoxCnt(int lev)	{ uint64 r = uint64(1) << mLogDim[lev]; return r*r*r; } // Gets the number of child nodes or child voxels of a node at the given level.
```
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

