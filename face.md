# 人脸


## 计算机动画方法：
| 年份 | 论文名 | 组织 | 输入/输出 | tracking方法 | fitting方法 | retarget方法 | 说明 |
| -- | --- | -- | ----- | ---------- | -------- | ---------- | -- |
| 00 | Pose Space Deformation | Centropolis | 输入：blendshapes + 一个sculpted的mesh/输出：RBF weights | ---------- | 定义: pose controls指任何操纵器； pose space指pose controls所延申的空间； 交互步骤：Sculpt pose， Define pose, Solve.| RBF网络及RBF训练，类似于MLP网络。 | 之前的方法skeleton supspace deformation(SSD), blendshape method or Shape interpolation；散点插值,scattered interpolation；具体case怎么解Sculpt形状到某个地方，要求解w. d^(x) = sum_k(wk(x)dk)/sum_k(wk(x)) vs d^(x) = sum_k(wk * phi(|x-xk|)) |
| 04 | Deformation Transfer for Triangle Meshes | MIT | 输入：Source blend shapes + target neutral shape / 输出：Target blend shapes | ---------- | -------- | ---------- | 开源实现[Deformation-Transfer](https://github.com/chand81/Deformation-Transfer), [deformation-transfer](https://github.com/Golevka/deformation-transfer); 公式简单，应该易于实现，不过下面一篇2017年的文章提到说deformation transfer在有些情况下效果比较差 |
| 06 | Animating Blendshape Faces by Cross-Mapping Motion Capture Data | USC | 输入：PCA系数+对应的Blendshape权重 / 输出：一个RBF模型系数，这样训练好后就直接拿RBF进行预测。 问题： RBF可否替换为MLP？ | ---------- | -------- | cross-mapping between motion capture data and target blendshape faces by RBF![faceanimation2](https://github.com/liangjin2007/data_liangjin/blob/master/faceanimation2.JPG?raw=true) | -- |
| 08 | Transferring the Rig and Animations from a Character to Different Face Models | -- | ----- | ---------- | -------- | RBF中添加了T(x) | -- |
| 11 | Real-time Avatar Animation from a Single Image | Saragih | 输入：图像 / 输出: ![图像](https://github.com/liangjin2007/data_liangjin/blob/master/realtime-capturing-from-single-image.JPG?raw=true)| ---------- | -------- | ---------- | 不用标定，看着质量不太好， 跟14 Kun.Zhou的目的一致 |
| 11 | Realtime Performance-Based Facial Animation | Saragih | -- | ---------- | -------- | GMM prior | -- |
| 12 | Facial animation retargeting framework using radial basis functions | Hungary | ----- | ---------- | -------- | 1.一种方式是将人脸上的marker点直接映射到target的bone上。 Maya Geometry Mapping: marker to bone mapping ![RBF](https://github.com/liangjin2007/data_liangjin/blob/master/RBF.JPG?raw=true) ; 2. 另一种是映射到blendshape；  | 输入xi，yi求解Rbf的w， 这样给定一个其他的x就可以从RBF函数求出对应的y |
| 13 | High Fidelity Facial Animation Capture and Retargeting With Contours | ILM | 输入:带marker点的视频流+脸部中性人脸+Blendshape+相机参数/输出： | 左右相机的2d marker tracking | 两个优化：1.根据2d,3d bundle, 眼睛轮廓，嘴巴轮廓求解blendshape；2.使用laplace变形+2d marker限制+2d curve限制+3d curves限制+3d point constraint共5项去生成corrective shape Delta V. | 分两步：1.迁移blendshape。即将actor的rig(指blendshape weights)直接用到creature上。需要使得creature的blendshape rig是从actor的blendshape用deformation transfer 迁移过去，最后直接使用actor的weights作用在creature上；2.迁移corrective shape， | 每帧处理时间需要1-2秒 |
| 13 | 3D Shape Regression for Real-time Facial Animation | MSRA/K.Zhou | 输入：视频流/输出：tracked mesh seq+avata animation | 分几步：1.预处理，训练一个模型使得可以从人脸图像预测出3d facial shape(注意：这里指的其实是3d landmarks), 具体训练方法为[2012]face alignment by explicit shape regression。2.预处理得到actor的blend shapes（文中有具体算法）。3.输入视频流，对每帧图像预测3d facial shape 4.![Dy method](https://raw.githubusercontent.com/liangjin2007/data_liangjin/master/faceanimation1.JPG) |两个考虑点：1.根据3d facial shape及user specific shape得到tracked mesh的head pose M及blendshape weight（类似于前一篇的bundle约束）2.构造一个GMM模型来增强时间连贯性，blendshape weight的动画先验信息，参考的是[2011]Realtime Performance-Based Facial Animation| 这一步跟前一篇文章一样，是通过最简单的方式将bs weights应用在avata上。 | 此方法跟Dynamixyz非常接近，研究此方法的缺陷也许可以找出Dynamicxyz中的不足。 相机外参是未知数, 省略了构造blendshape的过程及neutral mesh的过程（这部分可以节省大量的金钱及时间）；此方法中提到一些我曾经想实现的先验信息，比如pose先验，blendshape先验等。此方法论文中说可以实现24fps。|
| 13 | Online Modeling For Realtime Facial Animation | EPFL | 输入：RGBD+blendshapes / 输出：pose + blend shape weights + corrective| ---------- | -------- | corrective deformation fields by graph laplacian | 缺陷：深度的噪声 |
| 14 | Dynamic 2D/3D registration | -- | ----- | ---------- | -------- | ---------- | -- |
| 14 | Displaced Dynamic Expression Regression for Real-time Facial Tracking and Animation | K.Zhou | 输入：视频 / 输出：2d landmark + 3的facial shape + pose M | ---------- | -------- | ---------- | 打乒乓方法，主要贡献应该是减少了对不用用户的标定，比较快，可以实现28fps，其他作者侯启明记得是个大牛 |
| 15 | Dynamic 3D Avatar Creation from Hand-held Video Input ![](https://github.com/liangjin2007/data_liangjin/blob/master/faceanimation3.JPG?raw=true)| EPFL |预处理：通过多图重建neutral mesh和光照属性；Dynamic Modeing: 把neutral mesh变形到| ---------- | -------- | ---------- | multi-view stereo reconstruction from multiple images - > non-rigid registration -> optimized geometry, texture optimization, integrated texture -> textured model |
| 16 | Face2Face | Stanford etc | 输入: online source actor video + target video seq / 输出：a video with actor face | ![tracking](https://github.com/liangjin2007/data_liangjin/blob/master/facereenactment1.JPG?raw=true) | 两边都得到Pose M, expression, illumination及identity | expression transfer方法 | -- |
| 16 | Corrective 3D Reconstruction of Lips from Monocular Video | Disney/MaxPlanck/ETH Zurich | ----- | ---------- | -------- | ---------- | -- |
| 17 | Facial Retargeting with Automatic Range of Motion Alignment | KAIST | ----- | ---------- | -------- | 1.**改进的blendshape生成方法** 从character空间的face rig生成出actor空间的blendshape；<br/>2.加正则化L2惩罚大的weight， L1推导系数性； 惩罚时间改变penalization of temporal changes wf-1 - wf来减少jitter; 提出了新颖的geometry prior(提出了微分曲面属性上的操作来代替blendshapes权重上的操作，阻止几何假象) <br/>3.反向求解actor的blendshape利用parallel parameterization法做character animation| -- |
| 18 | State of the Art on Monocular 3D Face Reconstruction, Tracking, and Applications | -- | ----- | ---------- | -------- | parallel parameterization method | -- |
| 18 | Production-Level Facial Performance Capture Using Deep Convolutional Neural Networks | NVIDIA/Li Hao/USC | ----- | ---------- | -------- | ---------- | offline, multiple-view stereo |


## 知乎
人脸动作捕捉系统介绍https://www.zhihu.com/question/321811525

- 数据来源
  - 基于二维数据
  除数码相机之外，设备也可以是电脑摄像头、手机前置摄像头等移动设备上的摄像头，优点是成本低、易获取、使用方便，缺点是捕捉精度与其他方法相比较低。例子FaceRig。
  - 基于三维数据
  三维数据即在通过光学镜头获取二维数据的同时，通过一定的手段或设备，获取画面的深度。如相机阵列，结构光等。iPhoneX的人脸识别使用的是点阵投影器。

- 人脸拍摄环境下分为两种：
  - 有标记点
  如Vicon的Cara Post系统 
  - 无标记点
  如Mova, Dynamicxyz
  苹果推出的animoji使用iphoneX的前置摄像头驱动动画。

- 应用场景
  - 非实时应用场景
  - 实时应用场景
  实时应用通常带有展示性质，如 Vicon 与 Epic Games 合作展示的「Siren」形象，身穿动作捕捉套装和面部动作捕捉设备的演员可以即兴表演，三维「Siren」可以实时复制演员的动作。

## 参考资源
至于Markerless的方式，需要更加专业的相机和环境，一般实现非交互的效果，使用的是non-rigid registration and tracking algorithms。
[2010] High Resolution Passive Facial Performance Capture

- HelloFace https://becauseofai.github.io/HelloFace/
重点关注Face 3D, Face Capture, Face Lib&Tool
Face capture https://becauseofai.github.io/HelloFace/face_capture/

- Face-Tracking-Maya
  - Maya可使用commandPort命令来启动一个socket监听端口并执行接收到的字符串命令， 具体可以看FaceTrack.mel。http://help.autodesk.com/view/MAYAUL/2018/CHS/?guid=__Commands_commandPort_html
  - 这个例子是直接使用blob detector检测白色marker点并将marker点的移动映射到Maya中虚拟角色的joints的移动。直接发送移动数据给到Maya。
  
- 开源Maya rigging 插件，比较大的一个工程，可以学学Maya https://github.com/mgear-dev/mgear_dist  
  - 其中使用的Maya插件WeightDriver插件，底下使用了RBF solver去驱动权重，可以看一下代码怎么搞。

- 面部模型
3DMM representation
Probability Morphable Model http://gravis.dmi.unibas.ch/PMM/

- 人脸重建
[2017]Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression
Source code http://aaronsplace.co.uk/papers/jackson2017recon/


- 3dmm_cnn 
[2016]Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network
https://github.com/anhttran/3dmm_cnn
知乎介绍https://zhuanlan.zhihu.com/p/24316690
- vrn 
用CNN Regression的方法解决大姿态下的三维人脸重建问题
https://github.com/AaronJackson/vrn
- 4dface
人脸检测和从2d视频重建3d人脸
https://github.com/patrikhuber/4dface

- 头姿势估计
https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
  - Direct Linear Transformation
  - LM optimization
  
- 人脸对齐
2D和3D人脸对齐
https://github.com/1adrianb/face-alignment 
https://github.com/1adrianb/2D-and-3D-face-alignment

- 人脸识别
  - openface 一个基于深度神经网络的开源人脸识别系统，128D https://github.com/cmusatyalab/openface
  - OpenFace 人脸识别系统https://github.com/TadasBaltrusaitis/OpenFace
  -SeetaFaceEngine 人脸识别
    - https://github.com/seetaface/SeetaFaceEngine
    - https://www.zhihu.com/question/50631245

- 换脸
  - face_swap 换脸 https://github.com/YuvalNirkin
  - deepfakes_faceswap https://github.com/joshua-wu/deepfakes_faceswap

- 人脸表示
[2012]A Facial Rigging Survey

- Retargetting
[2017]Facial retargeting with automatic range of motion alignment

- 当前面部rigging工作流程
Registration
[2016][siga]-Modern Techniques and Applications for Real-Time Non-rigid Registration
ICP

- Action Unit识别
2016 CVPR https://github.com/zkl20061823/DRML

- 骨骼与皮肤的关系
https://github.com/tneumann/skinning_decomposition_kavan

- 深度学习方法
[2019][cvpr] Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set
微软https://github.com/microsoft/Deep3DFaceReconstruction
•	Python >= 3.5 (numpy, scipy, pillow, opencv)
•	Tensorflow >= 1.4
•	Basel Face Model 2009 (BFM09)
•	Expression Basis (transferred from Facewarehouse by Guo et al.)

[2019][cvpr] Monocular Total Capture: Posing Face, Body and Hands in the Wild
卡内基梅隆 https://github.com/CMU-Perceptual-Computing-Lab/MonocularTotalCapture

- SOTA https://github.com/YadiraF/PRNet
- python face3d https://github.com/YadiraF/face3d


## 基本视觉
- [1989]Motion and structure from two perspective views: algorithms, error analysis, and error estimation
已知图像上的2维坐标，如何算出3d

- [1998][IJCV] Determining the Epipolar Geometry and its Uncertainty : A Review
7点法算fundamental matrix
不知内参也不知外参，只能在像素坐标级别算

- [2004][PAMI] An efficient solution to the five-point relative pose problem
已知内参不知外参，称为Structure from Motion 问题， 5点法求解。

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

## 相关视觉重建 colmap
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

## 新闻
- AKAZE Tracker https://blog.csdn.net/Small_Munich/article/details/79418389
- VisualSFM http://ccwu.me/vsfm/
- Facebook人脸动作捕捉 http://finance.sina.com.cn/stock/relnews/us/2019-06-30/doc-ihytcitk8643416.shtml
- ILM人脸动补介绍 https://www.fxguide.com/fxfeatured/part-3-rogue-one-digital-humans/
  - Lightstage
  - FACS rig
- Weta https://www.fxguide.com/fxfeatured/weta-digitals-remarkable-face-pipeline-alita-battle-angel/
- Disney Meddusa FACS 两个作用，1.建expression shape library https://studios.disneyresearch.com/medusa/
- Dynamixyz 
  - http://www.dynamixyz.com/2019/03/13/unveiling-dynamixyz-stereo-hmc/
  - http://www.dynamixyz.com/2019/03/13/unveiling-dynamixyz-stereo-hmc/
  
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
- 单相机标定
  - EstimateAbsolutePose
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

## SFM
- SFM example https://github.com/nghiaho12/SFM_example/blob/master/src/main.cpp
- gtsam library https://github.com/borglab/gtsam/blob/develop/examples/SFMExample.cpp
 - 在所有图像对之间进行特征检测及匹配
 - Filter bad matches 
 - Recover motion between previous to current image and triangulate points
 
## Opencv标定
- 基本矩阵运算 https://blog.csdn.net/qq_29796317/article/details/73743296
- 标定及tracking https://www.cnblogs.com/mikewolf2002/p/5746667.html

## Face相关Paper笔记
- [2013] 3D Shape Regression for Real-time Facial Animation
  - 知识点：
    - 输入图像+landmark是可以轻松得到对应的3d fical shape。 底下的方法可以查看eos中根据face landmark拟合blendshape。
    - 同时还可以得到对应的3d landmark位置。
    - landmark可以手工进行修改。
    - 75 landmarks: 60 internal landmarks， 15 contour landmarks
    - 每个图像上的landmark对应于3d facial shape上的一个点。
  - Build User-specific 3D shape regressor
    - Get setup images + landmarks
      - performing a set of standard expressions to get images
      - facial landmark detecions
      - manually correction of landmarks
    - Construct user-specific Blendshapes
      即求解wid, wexp。额外还得到了各个相机的外参Mi。 
    
      - FaceWarehouse, 150 individuals, 46 FACS blendshapes for each.
      - bilinear face model Cr, Cr x wid x wexp
        - identity
        - FACS expression
      - 第一步，求解所有参数：wid跟wexp是未知数，类似于简单blendshapemarker点拟合（带约束二次规划），这个问题是非线性拟合，一般用LM算法求解，这边内参矩阵是已知的，外参不知，所以位置数为外参M, wid, wexp, 而且是相乘的关系。这边提到的解法为coordinate-descent。
        - 这边提示landmark是跟网格顶点对应的， 而不是三角形重心坐标。
      - 第二步，由于所有的图片都是同一个人的，那么对所有i, wid应该是一样的，Mi和wexp, i应该是不同的。 对所有的i优化一个目标函数             Ejoint = sum_i(sum_k||Q(Mi(Cr x wid x wexp, i) at vk) - ui(k)||^2)
      - 重复第一步和第二步。
      
    - 3D Facial Shape 恢复
      跟上一步同样一套数据setup images + landmarks, 再加上前一步得到的Mi, 这一步要得到作为训练数据的3d facial shape。可能跟blendshapeh很像。求解出来的是a即blendshape的权重。求解M及a， 通过coordinate-descent 方法。
      - El = sum_marker(||QM(B0+sum(a x Bi))-q||)， 求解a。
      - 正则化项 Ereg = ||  a - a* || 
      - POSIT算法求解rigid pose
      - gradient projection algorithm base on BFGS solver，来限定a的范围在0-1。
      - 每次迭代，更新vl为轮廓landmark vertex indices.
      - 得到M和a后通过M(B0+sum a x B)得到3d face mesh fi。
      - 从3d face mesh fi可以提取面部landmark点的3d坐标，这样就得到{S^o}i 称为3d facial shape 。
      
    
    
    
    - Construct training data for User-specific 3D Shape Regressor
      - 输入image, 2d landmark, user specific blendshapes, camera projection matrix, user specified blendshapes
      - 拟合blendshape系数使得blendshape of user specified blendshapes尽可能跟图片及2d landmark一致。
        
  - On-the-fly regression
    - User-specific Blendshape Generation
      - database FaceWarehouse: 150 individuals + 46 FACS blendshapes.
      - namely identity and expression, range-three core tensor Cr, 11k mesh vertices x 50 identity knobs x 47 expression         knobs.
      - two steps
        - 打乒乓方法
        - first step to solve [R t], wid, wexp
        - second step to refine wid.
        - 三次迭代就能收敛
    - Training Set Construction
      - 3D Faicial Shape Recovery
      - Data Augmentation
      
      
  - fit blendshapes from these labeled images
  - use blendshape model to calculate for each image its 3d facial shape composed of 3d landmark positions
  - train images and shapes
