# VolumeGVDB

## Topology
## Nodes
```
getCover: world-space covering size
getRange: index-space covering size
```
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


```
## Context
