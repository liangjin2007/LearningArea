# VolumeGVDB

## point cloud
```
point cloud Min, Max，p需要转成 p' = (p-Min)/(Max-Min)*65535这样的放到m_pnt1
wMin = Min
```
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
## Atlas/Channel
## Brick
## Aux
```
DataPtr mAux[MAX_AUX]; // 这个东西是在GPU上的。
Allocator *mPool;
void PrepareAux ( int id, int cnt, int stride, bool bZero, bool bCPU=false );

每个block是256个线程. // 是否可以Profile一下，什么值最合适。

AUX_PNTPOS 0
AUX_BOUNDING_BOX

```
## Context
