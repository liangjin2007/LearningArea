# VolumeGVDB

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
