
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
    

## FaceReconstruction
通过face landmark及Morphable Model去拟合。看例子包括如下功能：

- examples/face_landmark_detection_ex.cpp 由图像得到头部的pose，pose是用68个landmark表示。具体的实现方法是依赖于HOG特征+线性分类器+image pyramid+sliding window detection。具体实现paper为One Millisecond Face Alignment with an Ensemble of Regression Trees by Vahid Kazemi and Josephine Sullivan, CVPR 2014。使用的数据库为\biBUG 300-W face landmark dataset。用于学习。
```
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    


    This face detector is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset.  

    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

    


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/
```

- examples/fit-model.cpp 输入图片及landmarks。
- examples/webcam_face_fit_model_keegan.cpp
- examples/webcam_face_pose_ex.cpp
```
/**
 * This app demonstrates estimation of the camera and fitting of the shape
 * model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
 *
 * First, the 68 ibug landmarks are loaded from the .pts file and converted
 * to vertex indices using the LandmarkMapper. Then, an affine camera matrix
 * is estimated, and then, using this camera matrix, the shape is fitted
 * to the landmarks.
 */
```

- examples/scm-to-cereal.cpp
```
/**
 * Reads a CVSSP .scm Morphable Model file and converts it
 * to a cereal binary file.
 */
```

## openMVG
稀疏重建及获取相机参数，这一步在Dense Reconstruction之前属于Sparse Reconstruction

