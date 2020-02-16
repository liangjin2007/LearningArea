人脸
概述
人脸动作捕捉系统介绍https://www.zhihu.com/question/321811525

基于二维数据
除数码相机之外，设备也可以是电脑摄像头、手机前置摄像头等移动设备上的摄像头，优点是成本低、易获取、使用方便，缺点是捕捉精度与其他方法相比较低。例子FaceRig。
基于三维数据
三维数据即在通过光学镜头获取二维数据的同时，通过一定的手段或设备，获取画面的深度。如相机阵列，结构光等。iPhoneX的人脸识别使用的是点阵投影器。
人脸拍摄环境下分为两种：
有标记点
如Vicon的Cara Post系统 
无标记点
如Mova, Dynamicxyz
应用场景
非实时应用场景
电影、电视剧、游戏中的虚拟形象，动作捕捉完了需要较长时间调整以达到更好的效果
电视节目有时会使用相关技术在荧幕上演出虚拟形象
近期产生的虚拟偶像也是面部动作捕捉技术的应用之一
 
苹果推出的animoji使用iphoneX的前置摄像头驱动动画。
实时应用场景
实时应用通常带有展示性质，如 Vicon 与 Epic Games 合作展示的「Siren」形象，身穿动作捕捉套装和面部动作捕捉设备的演员可以即兴表演，三维「Siren」可以实时复制演员的动作。

面部捕捉系统各种架构
Quality low  high
Low
比如基于表演的面部动画
[2011] Realtime Performance-Based Facial Animation
RGBD摄像头
 


输入RGBD序列(Kinect)，输出对应帧的R t及表情权重等参数信息。
面部表情模型是用户表情空间的低维表示，通过预处理得到。怎么做呢？通用blendshape 
 





一般是基于单目的方法，输入的是RGBD或者RGB图像或者video。有许多研究在研究如何将quality不断的提高。

也有两类方法，具体可以参考[2018][EG] State of the Art on Monocular 3D Face
Reconstruction, Tracking, and Applications。
Analysis by Synthesis
这种是基于逆向渲染的思路，
Deep Learning Method




High
比如电影中使用的话，需要产生最好的avatar，那么在配置上就可能会比较麻烦，比如需要专门的studio环境，相机需要工业级相机等。

比如下面这个是带marker的论文：
[2005] Mirror MoCap: Automatic and efficient capture of dense 3D facial motion parameters from video
 

其架构如下：
 
这种方式，我的理解是，利用了通用head mesh的uv与marker之间的对应，记录了marker对应位置的运动，底下利用对极几何进行坐标对应。



至于Markerless的方式，需要更加专业的相机和环境，一般实现非交互的效果，使用的是non-rigid registration and tracking algorithms。
比如这篇[2010] High Resolution Passive Facial Performance Capture

 
相对的发射active light的基于结构光的方法
[2004]High 

HelloFace
https://becauseofai.github.io/HelloFace/
重点关注Face 3D, Face Capture, Face Lib&Tool
Face capture
https://becauseofai.github.io/HelloFace/face_capture/



面部模型
3DMM representation
Probability Morphable Model http://gravis.dmi.unibas.ch/PMM/

人脸重建

[2017]Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression
Source code http://aaronsplace.co.uk/papers/jackson2017recon/


3dmm_cnn 
[2016]Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network
https://github.com/anhttran/3dmm_cnn
知乎介绍https://zhuanlan.zhihu.com/p/24316690

vrn 
用CNN Regression的方法解决大姿态下的三维人脸重建问题
https://github.com/AaronJackson/vrn

4dface
人脸检测和从2d视频重建3d人脸
https://github.com/patrikhuber/4dface

人脸对齐
2D和3D人脸对齐
https://github.com/1adrianb/face-alignment 
https://github.com/1adrianb/2D-and-3D-face-alignment
人脸识别
openface 一个基于深度神经网络的开源人脸识别系统，128D
https://github.com/cmusatyalab/openface

OpenFace 人脸识别系统https://github.com/TadasBaltrusaitis/OpenFace

SeetaFaceEngine 人脸识别
https://github.com/seetaface/SeetaFaceEngine
https://www.zhihu.com/question/50631245

换脸
face_swap 换脸 https://github.com/YuvalNirkin
deepfakes_faceswap https://github.com/joshua-wu/deepfakes_faceswap


人脸表示
[2012]A Facial Rigging Survey



Retargetting
[2017]Facial retargeting with automatic range of motion alignment

当前面部rigging工作流程
?
Registration
[2016][siga]-Modern Techniques and Applications for Real-Time Non-rigid Registration
ICP

Action Unit识别
2016 CVPR https://github.com/zkl20061823/DRML

骨骼与皮肤的关系
https://github.com/tneumann/skinning_decomposition_kavan
News
Facebook人脸动作捕捉
http://finance.sina.com.cn/stock/relnews/us/2019-06-30/doc-ihytcitk8643416.shtml

初始工程框架

eos人脸morphable model https://github.com/patrikhuber/eos

[2019][cvpr] Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set
微软https://github.com/microsoft/Deep3DFaceReconstruction
•	Python >= 3.5 (numpy, scipy, pillow, opencv)
•	Tensorflow >= 1.4
•	Basel Face Model 2009 (BFM09)
•	Expression Basis (transferred from Facewarehouse by Guo et al.)


[2019][cvpr] Monocular Total Capture: Posing Face, Body and Hands in the Wild
卡内基梅隆 https://github.com/CMU-Perceptual-Computing-Lab/MonocularTotalCapture

SOTA https://github.com/YadiraF/PRNet

https://github.com/gabrielguarisa/facialMocap
https://github.com/justint/stringless\

python face3d https://github.com/YadiraF/face3d

# [1989]Motion and structure from two perspective views: algorithms, error analysis, and error estimation
- 已知图像上的2维坐标，如何算出3d

# [1998][IJCV] Determining the Epipolar Geometry and its Uncertainty : A Review
- 7点法算fundamental matrix
不知内参也不知外参，只能在像素坐标级别算

# [2004][PAMI] An efficient solution to the five-point relative pose problem
- 已知内参不知外参，称为Structure from Motion 问题， 5点法求解。

## 公式
```
- Notations

M=[x,y,z]
m=[u,v]
s[u,v,1]=P[x,y,z,1]
u,v是图像空间坐标（retinal image coordinates, 整数区间），不是normalized image coordinates。
P=A[R t]
图像空间坐标中的线l=[a,b,c], 对应的线的方程为au+bv+c=0。
符号距离为 d(m, l) = (au+bv+c)/sqrt(square(a)+square(b))

- Epipolar Geometry
C,C'
Plane PI, determined by C, C', M
epipole e, e'
epipolar line l, l'
epipolar constraint

```

# colmap
- 数据结构
```
Point2D
 Eigen::Vector2d xy_
 point3D_t point3D_id_

Point3D
 Eigen::Vector3d xyz_
 Eigen::Vector3ub color_
 double error_
 class Track track_
 
TrackElement
 image_id
 point2D_idx

Track
 std::vector<TrackElement> elements_

Image
 image_id_, name_, camera_id_, registered_, qvec_, tvec_, num_observations_,
 points2D_

Camera
 width_, height_, camera_id_, model_id_, params_
 ImageToWorld, WorldToImage

Correspondence
 image_id, point2D_idx

struct Image {
    // Number of 2D points with at least one correspondence to another image.
    point2D_t num_observations = 0;

    // Total number of correspondences to other images. This measure is useful
    // to find a good initial pair, that is connected to many images.
    point2D_t num_correspondences = 0;

    // Correspondences to other images per image point.
    std::vector<std::vector<Correspondence>> corrs;
};

struct ImagePair {
    // The number of correspondences between pairs of images.
    point2D_t num_correspondences = 0;
};
  
CorrespondenceGraph
 EIGEN_STL_UMAP(image_t, Image) images_;
 std::unordered_map<image_pair_t, ImagePair> image_pairs_;
 
Reconstruction
 cameras_
 images_
 points3D_
 image_pairs_
 
struct Problem {
    // Index of the reference image.
    int ref_image_idx = -1;

    // Indices of the source images.
    std::vector<int> src_image_idxs;

    // Input images for the photometric consistency term.
    std::vector<Image>* images = nullptr;

    // Input depth maps for the geometric consistency term.
    std::vector<DepthMap>* depth_maps = nullptr;

    // Input normal maps for the geometric consistency term.
    std::vector<NormalMap>* normal_maps = nullptr;

    // Print the configuration to stdout.
    void Print() const;
 };
 
 PatchMatch
 
 
 
```

- 代码例子
```
1.reconstruction_test.cc

void GenerateReconstruction(const image_t num_images,
                            Reconstruction* reconstruction,
                            CorrespondenceGraph* correspondence_graph) {
  const size_t kNumPoints2D = 10;

  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction->AddCamera(camera);

  for (image_t image_id = 1; image_id <= num_images; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(
        std::vector<Eigen::Vector2d>(kNumPoints2D, Eigen::Vector2d::Zero()));
    reconstruction->AddImage(image);
    reconstruction->RegisterImage(image_id);
    correspondence_graph->AddImage(image_id, kNumPoints2D);
  }

  reconstruction->SetUp(correspondence_graph);
}

- 两个相机
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, &reconstruction, &correspondence_graph);
```

- AKAZE Tracker https://blog.csdn.net/Small_Munich/article/details/79418389
- VisualSFM http://ccwu.me/vsfm/

## 知识点colmap
- Ransac
  - ComputeAlignmentBetweenReconstructions
    - LORANSAC<ReconstructionAlignmentEstimator, ReconstructionAlignmentEstimator> ransac(option);
    - ransac.Estimate
- TwoViewGeometry
  - EstimateCalibrated 
    - Estimate E, F, H
  - PoseFromEssentialMatrix(E, inlier_points1_normalized, inlier_points2_normalized, &R, &tvec, &points3D)
  - 
- BundleAdjuster
- RigBundleAdjuster
- ColorExtractor
- StereoFuser
- PoissonMesher
- DelaunayMesher
- PatchMatchStereo
- AutomaticReconstructor
  - Step1. SiftFeatureExtractor
  - Step2. Feature Matcher
    - Video数据类型： SequentialFeatureMatcher
    - INDIVIDUAL数据类型或者INTERNET数据类型： ExhaustiveFeatureMatcher或者VocabTreeFeatureMatcher
  - Step3. Sparse Mapper
    - SFM IncrementMapper
  - Step4. Dense Mapper
    - 1.COLMAPUndistorter
    - 2.Patch Match
    - 3.Stereo Fusion
      - mvs::StereoFusion
      - WriteBinaryPlyPoints(fused_path, fuser.GetFusedPoints());
      - mvs::WritePointsVisibility(fused_path + ".vis", fuser.GetFusedPointsVisibility());
    - 4.Poisson Meshing
      - mvs::PoissonMeshing
  - Result : Output depth_maps and normal_maps
  
- ExhaustiveMatcher
- SequentialMatcher
- SpatialMatcher
- TransitiveMatcher
- FeatureExtractor
- ImageRectifier
- ImageRegistrator
- ImageUndistorter
- ModelAligner
- ModelOrientationAligner
- ModelAnalyzer
- ModelConverter
- ModelMerger
- MatchesImporter
- Mapper
- HierarchicalMapper
- PointTriangulator
- VocabTreeBuilder
- VocabTreeMatcher
- VocabTreeRetriever

# opencv关于视觉重建的部分
- cvProjectPoints2
ProjectPoints2
投影三维点到图像平面

void cvProjectPoints2( const CvMat* object_points, const CvMat* rotation_vector,
                       const CvMat* translation_vector, const CvMat* intrinsic_matrix,
                       const CvMat* distortion_coeffs, CvMat* image_points,
                       CvMat* dpdrot=NULL, CvMat* dpdt=NULL, CvMat* dpdf=NULL,
                       CvMat* dpdc=NULL, CvMat* dpddist=NULL );
object_points
物体点的坐标，为3xN或者Nx3的矩阵，这儿N是视图中的所有所有点的数目。
rotation_vector
旋转向量，1x3或者3x1。
translation_vector
平移向量，1x3或者3x1。
intrinsic_matrix
摄像机内参数矩阵A：\begin{bmatrix}fx & 0 & cx\\ 0 & fy & cy\\ 0&0&1\end{bmatrix}
distortion_coeffs
形变参数向量，4x1或者1x4，为[k1,k2,p1,p2]。如果是NULL，所有形变系数都设为0。
image_points
输出数组，存储图像点坐标。大小为2xN或者Nx2，这儿N是视图中的所有点的数目。
dpdrot
可选参数，关于旋转向量部分的图像上点的导数，Nx3矩阵。
dpdt
可选参数，关于平移向量部分的图像上点的导数，Nx3矩阵。
dpdf
可选参数，关于fx和fy的图像上点的导数，Nx2矩阵。
dpdc
可选参数，关于cx和cy的图像上点的导数，Nx2矩阵。
dpddist
可选参数，关于形变系数的图像上点的导数，Nx4矩阵。
函数cvProjectPoints2通过给定的内参数和外参数计算三维点投影到二维图像平面上的坐标。另外，这个函数可以计算关于投影参数的图像点偏导数的雅可比矩阵。雅可比矩阵可以用在cvCalibrateCamera2和cvFindExtrinsicCameraParams2函数的全局优化中。这个函数也可以用来计算内参数和外参数的反投影误差。注意，将内参数和（或）外参数设置为特定值，这个函数可以用来计算外变换（或内变换）。

FindHomography

FindHomography
计算两个平面之间的透视变换

void cvFindHomography( const CvMat* src_points,
                       const CvMat* dst_points,
                       CvMat* homography );
src_points
原始平面的点坐标，大小为2xN，Nx2，3xN或者 Nx3矩阵（后两个表示齐次坐标），这儿N表示点的数目。
dst_points
目标平面的点坐标大小为2xN，Nx2，3xN或者 Nx3矩阵（后两个表示齐次坐标）。
homography
输出的3x3的homography矩阵。
函数cvFindHomography计算源平面和目标平面之间的透视变换H=\begin{bmatrix}h_{ij}\end{bmatrix}_{i,j}.

s_i \begin{bmatrix}x'_i \\ y'_i \\ 1\end{bmatrix}  \approx  H  \begin{bmatrix}x_i \\ y_i \\ 1\end{bmatrix}

使得反投影错误最小：

\sum_i((x'_i-\frac{h_{11}x_i + h_{12}y_i + h_{13}}{h_{31}x_i + h_{32}y_i + h_{33}})^2+          (y'_i-\frac{h_{21}x_i + h_{22}y_i + h_{23}}{h_{31}x_i + h_{32}y_i + h_{33}})^2)

这个函数可以用来计算初始的内参数和外参数矩阵。由于Homography矩阵的尺度可变，所以它被规一化使得h33 = 1

CalibrateCamera2
利用定标来计算摄像机的内参数和外参数
void cvCalibrateCamera2( const CvMat* object_points, const CvMat* image_points,
                         const CvMat* point_counts, CvSize image_size,
                         CvMat* intrinsic_matrix, CvMat* distortion_coeffs,
                         CvMat* rotation_vectors=NULL,
                         CvMat* translation_vectors=NULL,
                         int flags=0 );
object_points
定标点的世界坐标，为3xN或者Nx3的矩阵，这里N是所有视图中点的总数。
image_points
定标点的图像坐标，为2xN或者Nx2的矩阵，这里N是所有视图中点的总数。
point_counts
向量，指定不同视图里点的数目，1xM或者Mx1向量，M是视图数目。
image_size
图像大小，只用在初始化内参数时。
intrinsic_matrix
输出内参矩阵(A) \begin{bmatrix}fx & 0 & cx\\ 0 & fy & cy \\ 0&0&1\end{bmatrix}，如果指定CV_CALIB_USE_INTRINSIC_GUESS和（或）CV_CALIB_FIX_ASPECT_RATION，fx、 fy、 cx和cy部分或者全部必须被初始化。
distortion_coeffs
输出大小为4x1或者1x4的向量，里面为形变参数[k1, k2, p1, p2]。
rotation_vectors
输出大小为3xM或者Mx3的矩阵，里面为旋转向量（旋转矩阵的紧凑表示方式，具体参考函数cvRodrigues2）
translation_vectors
输出大小为3xM或Mx3的矩阵，里面为平移向量。
flags
不同的标志，可以是0，或者下面值的组合：
CV_CALIB_USE_INTRINSIC_GUESS - 内参数矩阵包含fx，fy，cx和cy的初始值。否则，(cx, cy)被初始化到图像中心（这儿用到图像大小），焦距用最小平方差方式计算得到。注意，如果内部参数已知，没有必要使用这个函数，使用cvFindExtrinsicCameraParams2则可。
CV_CALIB_FIX_PRINCIPAL_POINT - 主点在全局优化过程中不变，一直在中心位置或者在其他指定的位置（当CV_CALIB_USE_INTRINSIC_GUESS设置的时候）。
CV_CALIB_FIX_ASPECT_RATIO - 优化过程中认为fx和fy中只有一个独立变量，保持比例fx/fy不变，fx/fy的值跟内参数矩阵初始化时的值一样。在这种情况下， (fx, fy)的实际初始值或者从输入内存矩阵中读取（当CV_CALIB_USE_INTRINSIC_GUESS被指定时），或者采用估计值（后者情况中fx和fy可能被设置为任意值，只有比值被使用）。
CV_CALIB_ZERO_TANGENT_DIST – 切向形变参数(p1, p2)被设置为0，其值在优化过程中保持为0。
函数cvCalibrateCamera2从每个视图中估计相机的内参数和外参数。3维物体上的点和它们对应的在每个视图的2维投影必须被指定。这些可以通过使用一个已知几何形状且具有容易检测的特征点的物体来实现。这样的一个物体被称作定标设备或者定标模式，OpenCV有内建的把棋盘当作定标设备方法（参考cvFindChessboardCorners）。目前，传入初始化的内参数（当CV_CALIB_USE_INTRINSIC_GUESS不被设置时）只支持平面定标设备（物体点的Z坐标必须为全0或者全1）。不过3维定标设备依然可以用在提供初始内参数矩阵情况。在内参数和外参数矩阵的初始值都计算出之后，它们会被优化用来减小反投影误差（图像上的实际坐标跟cvProjectPoints2计算出的图像坐标的差的平方和）。

FindExtrinsicCameraParams2
计算指定视图的摄像机外参数
void cvFindExtrinsicCameraParams2( const CvMat* object_points,
                                   const CvMat* image_points,
                                   const CvMat* intrinsic_matrix,
                                   const CvMat* distortion_coeffs,
                                   CvMat* rotation_vector,
                                   CvMat* translation_vector );
object_points
定标点的坐标，为3xN或者Nx3的矩阵，这里N是视图中的个数。
image_points
定标点在图像内的坐标，为2xN或者Nx2的矩阵，这里N是视图中的个数。
intrinsic_matrix
内参矩阵(A) \begin{bmatrix}fx & 0 & cx\\ 0 & fy & cy \\ 0&0&1\end{bmatrix}。
distortion_coeffs
大小为4x1或者1x4的向量，里面为形变参数[k1,k2,p1,p2]。如果是NULL，所有的形变系数都为0。
rotation_vector
输出大小为3x1或者1x3的矩阵，里面为旋转向量（旋转矩阵的紧凑表示方式，具体参考函数cvRodrigues2）。
translation_vector
大小为3x1或1x3的矩阵，里面为平移向量。
函数cvFindExtrinsicCameraParams2使用已知的内参数和某个视图的外参数来估计相机的外参数。3维物体上的点坐标和相应的2维投影必须被指定。这个函数也可以用来最小化反投影误差。

Rodrigues2
进行旋转矩阵和旋转向量间的转换
int  cvRodrigues2( const CvMat* src, CvMat* dst, CvMat* jacobian=0 );
src
输入的旋转向量（3x1或者1x3）或者旋转矩阵（3x3）。
dst
输出的旋转矩阵（3x3）或者旋转向量（3x1或者1x3）
jacobian
可选的输出雅可比矩阵（3x9或者9x3），关于输入部分的输出数组的偏导数。
函数转换旋转向量到旋转矩阵，或者相反。旋转向量是旋转矩阵的紧凑表示形式。旋转向量的方向是旋转轴，向量的长度是围绕旋转轴的旋转角。旋转矩阵R，与其对应的旋转向量r，通过下面公式转换：

\theta \leftarrow norm(r)

r \leftarrow r/\theta

R = \cos(\theta)I + (1-\cos(\theta))rr^T + \sin(\theta) \begin{bmatrix}0&-r_z&r_y\\ r_z&0&-r_x\\ -r_y&r_x&0\end{bmatrix}

反变换也可以很容易的通过如下公式实现：

\sin(\theta) \begin{bmatrix}0&-r_z&r_y\\ r_z&0&-r_x\\ -r_y&r_x&0\end{bmatrix} = \frac{R-R^T}{2}

旋转向量是只有3个自由度的旋转矩阵一个方便的表示，这种表示方式被用在函数cvFindExtrinsicCameraParams2和cvCalibrateCamera2内部的全局最优化中。

Undistort2
校正图像因相机镜头引起的变形
void cvUndistort2( const CvArr* src, CvArr* dst,
                   const CvMat* intrinsic_matrix,
                   const CvMat* distortion_coeffs );
src
原始图像（已经变形的图像）。只能变换32fC1的图像。
dst
结果图像（已经校正的图像）。
intrinsic_matrix
相机内参数矩阵，格式为 \begin{bmatrix}fx & 0 & cx\\ 0 & fy & cy\\ 0&0&1\end{bmatrix}。
distortion_coeffs
四个变形系数组成的向量，大小为4x1或者1x4，格式为[k1,k2,p1,p2]。
函数cvUndistort2对图像进行变换来抵消径向和切向镜头变形。相机参数和变形参数可以通过函数cvCalibrateCamera2取得。使用本节开始时提到的公式，对每个输出图像像素计算其在输入图像中的位置，然后输出图像的像素值通过双线性插值来计算。如果图像得分辨率跟定标时用得图像分辨率不一样，fx、fy、cx和cy需要相应调整，因为形变并没有变化。

InitUndistortMap
计算形变和非形变图像的对应（map）
void cvInitUndistortMap( const CvMat* intrinsic_matrix,
                         const CvMat* distortion_coeffs,
                         CvArr* mapx, CvArr* mapy );
intrinsic_matrix
摄像机内参数矩阵(A) [fx 0 cx; 0 fy cy; 0 0 1].
distortion_coeffs
形变系数向量[k1, k2, p1, p2]，大小为4x1或者1x4。
mapx
x坐标的对应矩阵。
mapy
y坐标的对应矩阵。
函数cvInitUndistortMap预先计算非形变对应－正确图像的每个像素在形变图像里的坐标。这个对应可以传递给cvRemap函数（跟输入和输出图像一起）。

FindChessboardCorners
寻找棋盘图的内角点位置
int cvFindChessboardCorners( const void* image, CvSize pattern_size,
                             CvPoint2D32f* corners, int* corner_count=NULL,
                             int flags=CV_CALIB_CB_ADAPTIVE_THRESH );
image
输入的棋盘图，必须是8位的灰度或者彩色图像。
pattern_size
棋盘图中每行和每列角点的个数。
corners
检测到的角点
corner_count
输出，角点的个数。如果不是NULL，函数将检测到的角点的个数存储于此变量。
flags
各种操作标志，可以是0或者下面值的组合：
CV_CALIB_CB_ADAPTIVE_THRESH - 使用自适应阈值（通过平均图像亮度计算得到）将图像转换为黑白图，而不是一个固定的阈值。
CV_CALIB_CB_NORMALIZE_IMAGE - 在利用固定阈值或者自适应的阈值进行二值化之前，先使用cvNormalizeHist来均衡化图像亮度。
CV_CALIB_CB_FILTER_QUADS - 使用其他的准则（如轮廓面积，周长，方形形状）来去除在轮廓检测阶段检测到的错误方块。
函数cvFindChessboardCorners试图确定输入图像是否是棋盘模式，并确定角点的位置。如果所有角点都被检测到且它们都被以一定顺序排布（一行一行地，每行从左到右），函数返回非零值，否则在函数不能发现所有角点或者记录它们地情况下，函数返回0。例如一个正常地棋盘图右8x8个方块和7x7个内角点，内角点是黑色方块相互联通地位置。这个函数检测到地坐标只是一个大约地值，如果要精确地确定它们的位置，可以使用函数cvFindCornerSubPix。

DrawChessBoardCorners
绘制检测到的棋盘角点
void cvDrawChessboardCorners( CvArr* image, CvSize pattern_size,
                              CvPoint2D32f* corners, int count,
                              int pattern_was_found );
image
结果图像，必须是8位彩色图像。
pattern_size
每行和每列地内角点数目。
corners
检测到地角点数组。
count
角点数目。
pattern_was_found
指示完整地棋盘被发现(≠0)还是没有发现(=0)。可以传输cvFindChessboardCorners函数的返回值。
当棋盘没有完全检测出时，函数cvDrawChessboardCorners以红色圆圈绘制检测到的棋盘角点；如果整个棋盘都检测到，则用直线连接所有的角点。


姿态估计
CreatePOSITObject
初始化包含对象信息的结构

CvPOSITObject* cvCreatePOSITObject( CvPoint3D32f* points, int point_count );
points
指向三维对象模型的指针
point_count
对象的点数
函数 cvCreatePOSITObject 为对象结构分配内存并计算对象的逆矩阵。

预处理的对象数据存储在结构CvPOSITObject中，只能在OpenCV内部被调用，即用户不能直接读写数据结构。用户只可以创建这个结构并将指针传递给函数。

对象是在某坐标系内的一系列点的集合，函数 cvPOSIT计算从照相机坐标系中心到目标点points[0] 之间的向量。

一旦完成对给定对象的所有操作，必须使用函数cvReleasePOSITObject释放内存。

POSIT
执行POSIT算法
void cvPOSIT( CvPOSITObject* posit_object, CvPoint2D32f* image_points, 
              double focal_length,
              CvTermCriteria criteria, CvMatr32f rotation_matrix, 
              CvVect32f translation_vector );
posit_object
指向对象结构的指针
image_points
指针，指向目标像素点在二维平面图上的投影。
focal_length
使用的摄像机的焦距
criteria
POSIT迭代算法程序终止的条件
rotation_matrix
旋转矩阵
translation_vector
平移矩阵.
函数 cvPOSIT 执行POSIT算法。图像坐标在摄像机坐标系统中给出。焦距可以通过摄像机标定得到。算法每一次迭代都会重新计算在估计位置的透视投影。

两次投影之间的范式差值是对应点中的最大距离。如果差值过小，参数criteria.epsilon就会终止程序。

ReleasePOSITObject
释放3D对象结构
void cvReleasePOSITObject( CvPOSITObject** posit_object );
posit_object
指向 CvPOSIT 结构指针的指针。
函数 cvReleasePOSITObject 释放函数 cvCreatePOSITObject分配的内存。

CalcImageHomography
计算长方形或椭圆形平面对象(例如胳膊)的Homography矩阵
void cvCalcImageHomography( float* line, CvPoint3D32f* center,
                            float* intrinsic, float* homography );
line
对象的主要轴方向，为向量(dx,dy,dz).
center
对象坐标中心 ((cx,cy,cz)).
intrinsic
摄像机内参数 (3x3 matrix).
homography
输出的Homography矩阵(3x3).
函数 cvCalcImageHomography 为从图像平面到图像平面的初始图像变化(defined by 3D oblong object line)计算Homography矩阵。


对极几何(双视几何)
FindFundamentalMat
由两幅图像中对应点计算出基本矩阵

int cvFindFundamentalMat( const CvMat* points1,
                          const CvMat* points2,
                          CvMat* fundamental_matrix,
                          int    method=CV_FM_RANSAC,
                          double param1=1.,
                          double param2=0.99,
                          CvMat* status=NULL);
points1
第一幅图像点的数组，大小为2xN/Nx2 或 3xN/Nx3 (N 点的个数)，多通道的1xN或Nx1也可以。点坐标应该是浮点数(双精度或单精度)。:
points2
第二副图像的点的数组，格式、大小与第一幅图像相同。
fundamental_matrix
输出的基本矩阵。大小是 3x3 或者 9x3 ，(7-点法最多可返回三个矩阵).
method
计算基本矩阵的方法
CV_FM_7POINT – 7-点算法，点数目＝ 7
CV_FM_8POINT – 8-点算法，点数目 >= 8
CV_FM_RANSAC – RANSAC 算法，点数目 >= 8
CV_FM_LMEDS - LMedS 算法，点数目 >= 8
param1
这个参数只用于方法RANSAC 或 LMedS 。它是点到对极线的最大距离，超过这个值的点将被舍弃，不用于后面的计算。通常这个值的设定是0.5 or 1.0 。
param2
这个参数只用于方法RANSAC 或 LMedS 。 它表示矩阵正确的可信度。例如可以被设为0.99 。
status
具有N个元素的输出数组，在计算过程中没有被舍弃的点，元素被被置为1；否则置为0。这个数组只可以在方法RANSAC and LMedS 情况下使用；在其它方法的情况下，status一律被置为1。这个参数是可选参数。
对极几何可以用下面的等式描述:

p_2^T \cdot F \cdot p_1=0

其中 F 是基本矩阵，p1 和 p2 分别是两幅图上的对应点。

函数 FindFundamentalMat 利用上面列出的四种方法之一计算基本矩阵，并返回基本矩阵的值：没有找到矩阵，返回0，找到一个矩阵返回1，多个矩阵返回3。 计算出的基本矩阵可以传递给函数cvComputeCorrespondEpilines来计算指定点的对极线。

例子1：使用 RANSAC 算法估算基本矩阵。
int    numPoints = 100;
CvMat* points1;
CvMat* points2;
CvMat* status;
CvMat* fundMatr;
points1 = cvCreateMat(2,numPoints,CV_32F);
points2 = cvCreateMat(2,numPoints,CV_32F);
status  = cvCreateMat(1,numPoints,CV_32F);

/* 在这里装入对应点的数据... */

fundMatr = cvCreateMat(3,3,CV_32F);
int num = cvFindFundamentalMat(points1,points2,fundMatr,CV_FM_RANSAC,1.0,0.99,status);
if( num == 1 )
     printf("Fundamental matrix was found\n");
else
     printf("Fundamental matrix was not found\n");


例子2：7点算法（3个矩阵）的情况。
CvMat* points1;
CvMat* points2;
CvMat* fundMatr;
points1 = cvCreateMat(2,7,CV_32F);
points2 = cvCreateMat(2,7,CV_32F);

/* 在这里装入对应点的数据... */

fundMatr = cvCreateMat(9,3,CV_32F);
int num = cvFindFundamentalMat(points1,points2,fundMatr,CV_FM_7POINT,0,0,0);
printf("Found %d matrixes\n",num); 
ComputeCorrespondEpilines
为一幅图像中的点计算其在另一幅图像中对应的对极线。
void cvComputeCorrespondEpilines( const CvMat* points,
                                  int which_image,
                                  const CvMat* fundamental_matrix,
                                  CvMat* correspondent_lines);
points
输入点，是2xN 或者 3xN 数组 (N为点的个数)
which_image
包含点的图像指数(1 or 2)
fundamental_matrix
基本矩阵
correspondent_lines
计算对极点, 3xN数组
函数 ComputeCorrespondEpilines 根据外级线几何的基本方程计算每个输入点的对应外级线。如果点位于第一幅图像(which_image=1),对应的对极线可以如下计算 :

l_2=F \cdot p_1

其中F是基本矩阵，p1 是第一幅图像中的点， l2 - 是与第二幅对应的对极线。如果点位于第二副图像中 which_image=2)，计算如下:

l_1=F^T \cdot p_2

其中p2 是第二幅图像中的点，l1 是对应于第一幅图像的对极线，每条对极线都可以用三个系数表示 a, b, c:

a\cdot x + b\cdot y + c = 0

归一化后的对极线系数存储在correspondent_lines 中。




