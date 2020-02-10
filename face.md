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

