# 相机和图像
  - 针孔相机模型, 数字相机, 图像生成, 数字图像, 线性滤波(去噪，提取信息等), 高斯滤波，correlation and convolution
# 滤波器
  - scharr滤波器，用来计算图像差分，dx, dy表示差分阶数；拉普拉斯算子(Blob)，梯度的散度，sobel边检测算子(看着更美观)。
# 边检测
  - 法向不连续，深度不连续，颜色不连续，照明不连续, 用卷积实现导数，偏导数, 有限差分滤波, 图像梯度, 卷积的导数理论, Derivative of Gaussians, Laplace of Gaussians, 定位边 Canny's Edge Detector, sqrt(square(Gx)+square(Gy)), theta = atan2(Gy, Gx), 非极大抑制NMS, 训练边检测，Seam Carving(动态规划法求最小梯度能量路径)。
# 图像金字塔Image Pyramids
- 图像下采样， 走样问题（采样频率不够时出现），需要了解图像的结构，极小采样率Nyquist rate； 通过傅立叶变换找到信号的最高频率，设置Nyquist rate为此频率的两倍；高频率由尖锐的边导致，为了1/2下采样能更好的工作，需要降低原图像的最高频率，解决方法：Gaussian滤波来模糊图像，所以图像金子塔的构建过程是这样的，blur->subsampling->blur->subsampling->blur->subsampling...称为Gaussian Pyramids，图形学里称为mip map；
# 图像特征
- 图像局部或者全局的描述。不同的任务要选择不同的特征;提取哪里的特征？提取什么特征？特征如何匹配？；地标检测：不同的位置称为keypoints,不同的特征称为descriptors;跟踪tracking（场景或者人物在往哪里移动）：有特点的关键点及有特点的特征;视频中完全不同的帧的检测：全局特征
，descriptors可以简单，比如只用颜色；角色后的场景是什么类型；图片中两个杯子类型是否一样：每个杯子一个patch,一个patch一个descriptor, descriptor可以比较简单；在图片中找某个patch:每个位置匹配一下，比较像素值，仿射变换不变性；Detection（keypoints or densely）检测，Description描述符(空间范围及内容)，匹配；关键点keypoint检测,应用于图像stitching，[x,y,[x1,x2,...,xn]]，keypoint不能太少也不能太多，结构不清的patch不能检测，带有高对比度改变（梯度）容易检测，直线段不好检测，有两个方向的梯度最容易检测(corners)；Corner的特点是某点（x,y）处的patch往任何方向移动的时候，亮度变化都很大（注意不是说取不同方向的像素时，亮度差值都很大）;如何建模patch的移动呢: Harris Corner Detector。 计算Ix, Iy, 计算Ix^2, Iy^2, Ix.Iy， 计算M， 计算Cornerness Score， R=det(M)-alpha trace(M), 找到那些R大于某个threshold的点，非极大抑制得到局部极大的点; 旋转、旋转不变，但不是Scale不变。[WSSD](https://github.com/liangjin2007/data_liangjin/blob/master/wssd1.jpg?raw=true)
- 伸缩不变特征点检测：Scale Invariant Interest Point Detection; Lowe's DoG scale invariant interest points 计算高斯图像金字塔，计算高斯差(是对LoG的近似)， 计算局部极值according to (x,y,octave), one octave包含5张图片，下一个octave在空间上比前一个octave尺寸变成一半。
- 图像局部描述符 SIFT,HOG,GLOH，SURF,DAISY, LBP, Shape Contexts, Color Histograms...；SIFT 位置，scale, 主要定向，128维特征；
- 在新的视点匹配平的对象：1.仿射变换 Scale， Rotation, Shear， Mirror变换， ；线性变换：原点映射到原点，线映射到线，平行线仍然是平行的，比率保留下来，多个线性变换相乘仍然是线性变换；仿射变换，线性变换+平移，原点不一定映射到原点，线映射到线，平行线仍然平行，比率保留，封闭，矩形变成平行四边形；计算仿射变换，找到匹配点，去掉outlier, 最小二乘; RANSAC, 从最少采样点开始，拟合直线，记数inliers数目，重复这个过程，选择inliers数最多的参数。 2.Homography 平行线不再平行以及比率不保持，矩形变成四边形，仿射变换是Homography的特殊形式，物体较远时可以用仿射变换替代homography;什么是Homography, 把四边形变换为矩形; 全景缝合： Homography变换，然后用Laplacian Pyramid Blending 去掉缝。
- Visual Word 可视单词
  - Each image contain several local feature descriptors(e.g. SIFT), array of 128-dim vectors
  - Collect all the SIFTS of all the pictures
  - K-Means Cluster Fine K cluster centers named as W1,W2,...,WK.
  - Build Inverted File Index逆文件索引
    - W1: pic1, pic2, picn
    - W2: pic3, pic6
    - ...
  - Bag Of Visual Words
    - histogram of visual words
    - term frequency-inverse document frequency (tf-idf)
- CNN Feature
# 相机模型
相机与人眼，针孔相机模型；光圈aperture,光圈太小会出现衍射效果diffraction；镜片把光线汇聚到film上，特殊距离焦距；成像Imaging，真实世界的2d投影，捕获两类信息：几何的和测光的，复杂的3D-2D关系，相机模型近似这些关系；投影矩阵，透视投影，正交投影，虚像，具体请看计算机图形学中关于投影矩阵的推导。
# Homography
transpose([wx,wy,w]) = P transpose([X,Y,Z,1]) = K[R | t] transpose([X,Y,Z,1])；透视投影特点：所有平行线相交于1点，称为消失点；Stereo 立体照片；从单张图片获取深度，shape from shading， shape from texture, 遮挡可以给我们深度的提示，从Google街景Z-Buffer中获得深度，然后绘制比较酷的场景； 来自两个View的深度：Stereo， 极线几何，并行标定相机， 输入左右两张图片，获取视差映射，在扫描线上匹配xl和xr（使用SSD或者normalized correlation（即内积））， 更大的patch，视差映射更光滑，Z = f T/(xr-xl)。[disparity map](https://github.com/liangjin2007/data_liangjin/blob/master/disparity_map.jpg?raw=true); Superpixels.
# 识别
图片内容是什么地标？ 场景分类（室内/室外，城市，森林，工厂，etc），分类(对象是人，车，etc)， 图像标注（街道，人物，室内），检测（找到某个特定类别的物体，比如行人），分割（哪些像素属于某个类型），姿势估计（每个对象的姿势），属性识别（估计对象颜色，尺寸，等）,商业化（推荐应该穿什么衣服才能找到女朋友），动作识别（图像中正在发生什么？是不是在步行），监控（为什么那两个人跪着）; template matching模版匹配（normalized correlation内积）， 相似性提取(以图搜图)；识别很困难，因为遮挡，伸缩，变形，背景混乱，照明，视点，对象姿势，类内变化很大，类别太多（约有10000到30000），以神经网络输出特征为描述符，比SIFT要好； 检测，找到感兴趣的物体； RCNN， Selective Search, Superpixels, PASCAL VOC challenge 20 object classes, 1万图像for training， 1万for testing。
# 检测
- interest point 基于兴趣点的方法：比如Harris Corners, 然后Hough voting; 找到图片中的直线：Hough Transform;找到图片中的圆；Hough Voting for一般形状
- Hough Transform
  - 图像空间 与 Hough（参数）空间之间的关系
    - 图像空间中的直线对应于Hough空间中的一个点
    - 图像空间中的一个点给经过这个点的所有直线投票，经过这个点的所有直线在Hough空间中是一条直线。即每个点对应于Hough空间中一条直线。
    - Hough空间中两条直线相交的点对应于图像空间中一条直线
  - 用每个图像点进行vote
  - 在Hough空间中找到peaks，每个peak在图像中是一条直线。
  - Hough空间会进行量化（离散化），用计数的方式进行voting。
  - [hough voting algorithm](https://github.com/liangjin2007/data_liangjin/blob/master/hough_voting.jpg?raw=true)
  - 一般化的Hough Voting。
- Mean Shift
- 滑动窗口方法 sliding windows
构建图像金字塔，扫描图像(location, scale)，提取窗口特征，在每个位置跑SVM分类器，融合多次检测（location, scale）， 带包围盒的对象检测； 提取特征HOG（直方图of梯度）；二分类问题；找到很多有重叠区域的bounding box， 非极大抑制NMS（每次选择分数最高的bounding box，删掉跟它重合度超过某个界限的boundingbox）
- DPM Method
- Generate region proposals
  - 分割算法， Graph based Segmentation, SLIC, SEEDS, Normalized Cuts, gPb, Hierarchical Graph-based Video Segmentation, Temporal Superpixels
# 分类
- SVM
- CNN
- Softmax
- L-Softmax
# 最新物体检测
- [RCNN](https://web.njit.edu/~usman/courses/cs698_spring18/RCNN.pdf)
![两步法](https://github.com/liangjin2007/data_liangjin/blob/master/RCNN.jpg?raw=true)， selective search, segment image into 1k or 2k regions, similarity computation, merge similar regions;  similarity: color similarity, 颜色直方图, texture similarity, 高斯梯度直方图; 

# 语义分割
# 实例分割
# 机器人
- 折衣服 https://www.youtube.com/watch?v=gy5g33S0Gzo
# 自动驾驶领域
- https://www.youtube.com/watch?v=4fxFDypHZLs
# 补洞
# 从单张图片制作视频
- http://www.cs.cmu.edu/~om3d/
# 可视化和游戏分析
# 电影特效
- https://www.youtube.com/watch?v=WDwTQ57YyzI#t=21
# 重建3D世界
- https://www.youtube.com/watch?v=IgBQCoEfiMs
# 指出人们穿着什么
- http://clothingparsing.com
# 检测和分析人脸
- http://www.rekognition.com/
# 美化人脸
# 指纹识别
# 法医鉴定技术
# 对象识别
# 识别电影海报
# 3D Pose Estimation with Depth Sensors
# 面部表情识别
# 面部动作识别
# 身体形变
# 头部分割
# 手势动作
# 手势轨迹
# 身体姿态
# 美颜
- 2d特征点检测，变换到3d, 变形，映射回去，然后三角剖分，变形。
# 成像质量
- USM
- HDR
# 编解码优化
# 人脸形变
# 人脸检测
# 人脸分割
# 人脸关键点检测
# 人脸属性
# 人脸动作
# 人像分割
- 深度学习+传统matting
# 头发分割
# 手势识别
- FasterRCNN
# 姿态识别
- OpenPose
# 视觉跟踪
# 场景分类
# 图像处理
# SLAM
# 视频码率优化
# 蒸馏学习
# 模型压缩AliNN
