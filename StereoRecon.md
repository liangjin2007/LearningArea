
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

## colmap流程
COLMAP is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline with a graphical and command-line interface. It offers a wide range of features for reconstruction of ordered and unordered image collections. The software is licensed under the new BSD license. 

- https://demuc.de/colmap/
- https://colmap.github.io/

SFM从图片做稀疏重建和求解相机矩阵, MVS的输入都是image+pose。 https://colmap.github.io/tutorial.html

- colmap有图形界面和命令行两种方式
- colmap数据存成了SQlite格式 对我们这种少量数据的模型来说，不需要用SQlite
- Camera Model
- Output Format
  - Sparse Reconstruction
    - cameras.txt
    - images.txt
    - points3D.txt
    - read脚本scripts/python/read_model.py， scripts/matlab/read_model.m
  - Dense Reconstruction
    - images
    - sparse
      - cameras.txt
      - images.txt
      - points3D.txt
    - stereo
      - consistency_graphs
        - image1.jpg.photometric.bin
      - depth_maps
        - image1.jpg.photometric.bin
      - normal_maps
        - image1.jpg.photometric.bin
      - patch-match.cfg
      - fusion.cfg
    - fused.ply
    - meshed-poisson.ply
    - meshed-delaunay.ply
    - run-colmap-geometric.sh
    - run-colmap-photometric.sh
    


