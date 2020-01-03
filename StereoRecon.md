
## openMVS流程
可以看openMVS_sample/MvgMvs_Pipeline.py开头部分关于使用的说明。它是一个输入一堆图片输出3dmesh+texture的一个流程。主要步骤如下：
```
#
# this script is for easy use of OpenMVG and OpenMVS
#
#usage: SfM_MyPipeline.py input_dir output_dir
#                         [-h] [-f FIRST_STEP] [-l LAST_STEP] [--0 0 [0 ...]]
#                         [--1 1 [1 ...]] [--2 2 [2 ...]] [--3 3 [3 ...]]
#                         [--4 4 [4 ...]] [--5 5 [5 ...]] [--6 6 [6 ...]]
#                         [--7 7 [7 ...]] [--8 8 [8 ...]] [--9 9 [9 ...]]
#                         [--10 10 [10 ...]] [--11 11 [11 ...]]
#
#Photogrammetry reconstruction with these steps :
#	0. Intrinsics analysis	openMVG_main_SfMInit_ImageListing
#	1. Compute features		openMVG_main_ComputeFeatures
#	2. Compute matches		openMVG_main_ComputeMatches
#	3. Incremental reconstruction	openMVG_main_IncrementalSfM
#	4. Colorize Structure	openMVG_main_ComputeSfM_DataColor
#	5. Structure from Known Poses	openMVG_main_ComputeStructureFromKnownPoses
#	6. Colorized robust triangulation	openMVG_main_ComputeSfM_DataColor
#	7. Export to openMVS	openMVG_main_openMVG2openMVS
#	8. Densify point cloud	OpenMVS/DensifyPointCloud
#	9. Reconstruct the mesh	OpenMVS/ReconstructMesh
#	10. Refine the mesh		OpenMVS/RefineMesh
#	11. Texture the mesh	OpenMVS/TextureMesh
```

可以看到
