# tutorials
http://cvpr2018.thecvf.com/program/tutorials
http://cvpr2019.thecvf.com/program/tutorials
...


# 视觉历史
1959开始研究猫脑中哪些部分影响视觉， 1963年Block World， 1965年第一个暑期视觉项目， 1970年出视觉方面的第一本书David Marr，处于视觉表示阶段， 1979年Generalized Cylinder, 1973年Pictorial Structure, 1987年边检测，1997年Normalized Cut, 2001年Face Detection Viola & Jones, 1999年SIFT&对象检测， 2006年Spatial Pyramid Matching， 2005年HoG, 2009年DPM, 2006-2012， PASCAL Visual Object Challenge, 2009年ImageNet, 2015年Image Classification Challenge
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
# 图像分类
Nearest Neighbor， K-Nearest Neighbors, majority vote from K closest points, L1 曼哈顿距离Manhattan, L2欧氏距离；训练集，验证集，测试集；交叉验证，k-fold交叉验证；线性分类器f(x)=Wx, 代数视角，可视化视角，几何视角； 损失函数Loss Function, 总体上是每个样本上损失函数的和；每个样本的损失函数， 多类SVM loss（线性SVM）， 正则化， L2正则化，正则化倾向于更简单的模型；Softmax分类（多项逻辑回归）， 理解粗糙的分类分数为概率， P(Y=k|X=xi)=e^sk/sum(e^sj), 预测条件概率， KL散度，交叉熵;优化方法，随机搜索，跟随斜率，梯度下降，随机梯度下降（minibatch近似）；神经网络， 卷积层公式(W-F+2P)/S+1, W原来宽度，F filter尺寸， P padding, S stride; 池化层，(W-F)/S+1； 训练神经网络，数据预处理，激活函数，Batch Normalization for 激活函数（zero-mean, unit variance）,选择网络架构，看损失函数是否合理；CNN， AlexNet, ZFNet, VGG, GoogLeNet, Inception, ResNet, Network in Network(NiN), Wide ResNet, ResNetXt, Stochastic Depth, Good Practices for Deep Feature Fusion;

# 图像标注和回归神经网络
图像Captioning，输入图像给出一系列单词，一系列单词->情感，机器翻译；RNN计算图，多对多（机器翻译，帧级视频分类），多对1(情感分类， 编码一个序列为单独的向量)， 1对多（图像标注），顺序处理非序列数据;某个时间步上的输入向量，老状态，新状态;  字符级语言模型; 向后Backpropagation,truncated back propagation; 搜索可理解的cell；图像标注；带注意力机制的图像标注；
[prepare-photo-caption-dataset-training-deep-learning-model](https://machinelearningmastery.com/prepare-photo-caption-dataset-training-deep-learning-model/) ;Dataset Flickr8k_Dataset.zip, Flickr8k_text.zip;

# 检测和分割
- 语义分割（对象及背景）
每个pixel打个标签，不区分实例；滑动窗口法，分类patch中心;方法2，全卷积(same)；学习反卷积，下采样，In-netwrk上采样(Unpooling, Max-Unpooling)，Learnable Upsampling(转置卷积,也称为-Deconvolution (bad) -Upconvolution -Fractionally strided convolution -Backward strided convolution) ; Multi-view 3D 重建 ：2D下采样（编码器）-> 3D卷积LSTM -> 3D上采样（解码器）-> 3D softmax loss；
- 分类和定位（一个对象）
看成两个任务，定位看成一个回归问题，一个Softmax loss，一个L2 loss； 即多任务loss；前面部分用迁移学习；
- 对象检测（多个对象）
每个图像需要有不同数目的输出；把对象检测看成分类;
  - Proposal类方法
  滑动窗口，需要运用很多CNN到子窗口上，huge number of locations, scales and aspect ratio；Region Proposal/Selective Search; R-CNN: 从某种proposal方法得到Region of Interest->CNN on ROI -> bbox回归+SVM ； Fast R-CNN： 整个图像作为卷积网络输入->从某一层进行Proposal得到ROI->RoiPooling层->Softmax+bbox回归; Faster R-CNN： Region Proposal Network(RPN)预测proposals from features;
  ![两步法](https://github.com/liangjin2007/data_liangjin/blob/master/RCNN.jpg?raw=true)， selective search, segment image into 1k or 2k regions, similarity computation, merge similar regions;  similarity: color similarity, 颜色直方图, texture similarity, 高斯梯度直方图; 
  - 非Proposal类方法
  YOLO： Divide image into 7x7 grid-> 每个格子里面回归 dx_ci,dy_ci,dh_ci,dw_ci,confidence_ci->输出7x7x(5*B+C)
  YOLO2:Better and Faster
  SSD: Slower but more accurate than YOLO; Faster but less accurate than Faster R-CNN
  MASK R-CNN: Region Proposal Network(RPN), Region of interest feature alignment RoiAlign
  Seg-Net: Encoder-Decoder framework
  UNet
- 实例分割（多个对象）
Mask R-CNN: 分割姿势
- 密集标注
- 可视化基因
- 场景图生成
- 3D对象检测
  - 单个相机
  - 相机+雷达
  - Faster RCNN in 3D
- RGB-Depth相机
- 点云体素化

# 生成模型
监督学习：->
无监督学习：K-means Clustering; PCA 降维;  t-SNE; Autoencoders自编码器； 特征学习；密度估计; 
生成模型：给定训练数据的概率分布，想要学习一个近似于训练数据分布的近似分布，并用这个近似分布去生成数据。处理的是密度估计问题;生成模型的why? 时序数据的生成模型可以用来模拟和计划,etc;[生成模型分类](https://github.com/liangjin2007/data_liangjin/blob/master/generative_model.jpg?raw=true);
- 显式模型，易处理密度，能直接预测极大似然值
  - Fully visible belief network
    完全可视置信度网络，显示密度模型，将图像表达为似然函数并用链式法则分解为逐像素的条件概率。
    p(x;theta) = Product(p(xi|x0,...xi_1;theta))
  - PixelRNN
    从图像角落开始生成像素，依赖于之前的以RNN建模的像素;缺点是非常慢
  - PixelCNN
  从图像角落开始生成像素，依赖于之前的以CNN建模的像素; 训练极大似然估计； 似然函数=联合密度函数；比PixelRNN要快，但是仍然慢。
- 近似密度估计
  - VAE
# 变分自编码器 VAE
从无打标训练数据学习一个低维度的特征表示；从线性，非线性，到深度全连接，到ReLU CNN; 想要用特征来捕获数据里有意义的变化；编码，解码；图像分类中的CNN可以看成一个编码器；特征捕获了训练集中的变化因素；与自编码器的区别：自编码器学习的隐藏特征能尽量重现输入图像，而变分自编码器能重建输入数据的模型情况,假设训练数据是从latent z重建出来的，假设z服从某个先验分布，比如gaussian；直觉：x是输入图像，z是隐藏因素，用来重建x：属性，定向等； p(z;theta)~guassian, p(x|z;theta)比较复杂，用神经网络表示；具体看cs231n_2018_lecture12.pdf中的具体推导，如何去解决训练复杂的似然函数p(x;theta) = Integrate(p(z;theta)p(x|z;theta), dz)的问题。 另外可以查看trello上已完成的autoencoder. https://trello.com/b/bwqk2uTp/jinl-roadmap

# GAN
两个玩家的游戏，生成网络，判别网路, minmax game；
- the GAN zoo https://github.com/hindupuravinash/the-gan-zoo
- the coolest GAN is from NVIDIA

# 可视化和理解深度神经网络
- 第一层filter/kernel的可视化
再高层的可视化没有那么令人感兴趣
- 最后一层特征层的可视化
  - knn最近邻（l2范数）; 降维到二维： 包括PCA，t-SNE； 可视化激活： conv5 feature map 128x13x13可视化成128个13x13的图像； 
- Maximally activating patches: 选择一个层，选择一个通道(channel)，用网络预测许多图片，记录预测的通道上的值，visualize image patches that corresponding to maximal activations；
- 显著性：用遮挡的方法去看盖住哪些像素对预测概率影响最大；
- 显著性映射saliency map：通过BP
  compute gradient of unnormalized class score with respect to image pixels, take absolute value and max over RGB channels; Saliency Maps: 用来做分割， use graphcut on saliency map;  
- Intermediate Features via (guided) backprop
  具体做法：Pick a single intermediate neuron, e.g. one value in 128 x 13 x 13 conv5 feature map, Compute gradient of neuron value with respect to image pixels, Guided BP是用来找到图像中对应于某个神经元的区域。
- 可视化CNN特征：Gradient Ascent
  产生一张合成的图像，这张图像会最大化某个神经元。
- 欺骗图像/对抗例子
  具体做法：(1) Start from an arbitrary image，(2) Pick an arbitrary class， (3) Modify the image to maximize the class， (4) Repeat until network is fooled
- DeepDream：增强存在的特征
  增强网络中某一层的神经元激活；具体做法： Choose an image and a layer in a CNN; repeat：1. Forward: compute activations at chosen layer 2. Set gradient of chosen layer equal to its activation 3. Backward: Compute gradient on image 4. Update image
- 特征反转feature inversion
输入特征，找一张图使得它的特征尽可能跟给定特征相近且自然， 正则化使用Total Variation regularizer，使得空间上更光滑。
- 纹理合成texture synthesis
输入一个小纹理，产生一个大纹理。
- 神经纹理合成neural texture synthesis
每一层计算一个Gram矩阵，大小为CxC, 做Gram重建； Gram Reconstruction; Texture=Artwork
```
1. Pretrain a CNN on ImageNet (VGG-19)
2. Run input texture forward through CNN, record activations on every layer; layer i
gives feature map of shape Ci × Hi × Wi
3. At each layer compute the Gram matrix
giving outer product of features: (shape Ci × Ci)
4. Initialize generated image from random noise
5. Pass generated image through CNN, compute Gram matrix on each layer
6. Compute loss: weighted sum of L2 distance between Gram matrices
7. Backprop to get gradient on image
8. Make gradient step on image
9. GOTO 5
```
- 神经风格转化 neural style transfer
feature+Gram Reconstruction; 问题：需要Forward/BP很多步，非常慢。
- Fast style transfer

# 弱监督学习



# 机器人
- 折衣服 https://www.youtube.com/watch?v=gy5g33S0Gzo
# 自动驾驶领域
- https://www.youtube.com/watch?v=4fxFDypHZLs
# 低秩矩阵恢复

# 补洞
```
- Patch based inpainting, synthesize texture by collecting small image patches
- [2014]Learning inpainting with GAN
- [2016]Learning-based inpainting
- [2017][siggraph]Globally and Locally Consistent Image Completion
  - Completion Network
  - Discriminator Network
    - Local
    - Global
    
```
- https://github.com/tadax/glcic

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
# 图像全景生成
- ImageStitch方法：[2007]AutoStitch, [2013]APAP, [2014]SPHP + APAP, [215]AANAP, 

# 老照片修复
https://docs.google.com/presentation/d/14VL0wPYdZIuOWYzobaYvf3Y_LnnijEuD2tCy4h8utjw/edit#slide=id.gffed9d51e_1_21
问题：折叠，缺失，曝光和颜色协调性问题
  - 算法流程
    - 找出问题区域
    - 分割图像： No Face or Face
    - No Face
      - 小的折叠或者洞-> Inpainting
      - 大的洞 -> fragment based image completion
              -> scene completion
    - Face
      - Non-fine-grained features(cheeks, forehead):
        - Small (Image Inpainting [4])
        - Patch (Segmented-based image completion[8])
      - Fine-grained features (Missing patches):
        - Graph Laplace for Occluded Face Completion[6][7]
        - Stronger conditional Generative Adversarial Network [5]
# 场景分类
# 图像处理 
- 图像增强Image Enhancement [2015][TOG]Automatic Photo Adjustment Using Deep Neural Networks https://sites.google.com/site/homepagezhichengyan/home/dl_img_adjust
  - 自动风格化和色调调整
  - 前景突出效果
  - Local Xpro effect
  - 水彩效果
  - 具体做法：语义分割
# SLAM
# 视频码率优化
# 蒸馏学习
# 模型压缩AliNN
# 房间重建
[2009]Recovering the Spatial Layout of Cluttered Rooms
[2014][cg]Automatic room detection and reconstruction in cluttered indoor environments with complex room layouts
- 提取平面墙->将它们与背景分离->处理缺失数据->提取独立房间(使用diffusion process)
- robust statistics方法
- 关键词 room reconstruction, room layout reconstruction
[2016][cvpr]Efficient 3D Room Shape Recovery from a Single Panorama
- 从单张全景图重建room的形状
# 图像合成
- 合成Sketches
```
[2017][cvpr]Scribbler: Controlling Deep Image Synthesis with Sketch and Color, CS4476.szeliski
```
# 图片美学
- paper列别
```
[2004]Classification of Digital Photos Taken by Photographers or Home Users.pdf
[2004][ijcv]Distinctive Image Features from Scale-Invariant Keypoints.pdf
[2006]Graph-Based Visual Saliency.pdf
[2006][ECCV]Studying Aesthetics in Photographic Images Using a Computational Approach.pdf
[2007]Learning Visual Attributes.pdf
[2007][cvpr]Learning to Detect A Salient Object.pdf
[2008]Algorithmic inferencing of aesthetics and emotion in natural images- An exposition.pdf
[2008][eccv]Photo and Video Quality Evaluation- Focusing on the Subject.pdf
[2009]Sensation-based Photo Cropping.pdf
[2009][cvpr]Describing Objects by their Attributes.pdf
[2009][cvpr]Learning To Detect Unseen Object Classes by Between-Class Attribute Transfer.pdf
[2010]A Framework for Photo-Quality Assessment and Enhancement based on Visual Aesthetics.pdf
[2010]ACQUINE- Aesthetic Quality Inference Engine – Real-time Automatic Rating of Photo Aesthetics.pdf
[2011]Assessing the aesthetic quality of photographs using generic image descriptors.pdf
[2011]Clustering-An Analysis of Single-Layer Networks in Unsupervised Feature Learning.pdf
[2011][CVPR]Aesthetic Quality Classification of Photographs Based on Color Harmony.pdf
[2011][CVPR]High level describable attributes for predicting aesthetics and interestingness.pdf
[2011][ICCV]Content-Based Photo Quality Assessment.pdf
[2011][cvpr]Learning to Predict the Perceived Visual Quality of Photos.pdf
[2012]AVA Dataset.pdf
[2012][cvpr]Meta-Class Features for Large-Scale Object Categorization on a Budget.pdf
[2013]Learning beautiful (and ugly) attributes.pdf
[2013]Size Does Matter- How Image Size Affects Aesthetic Perception.pdf
[2014]RAPID--Rating Pictorial Aesthetics using Deep Learning.pdf
[2014]Recognizing image style.pdf
[2014][ijcv]Discovering beautiful attributes for aesthetic image analysis.pdf
[2015][iccv]Deep Multi-Patch Aggregation Network for Image Style, Aesthetics, and Quality Estimation.pdf
[2016]A Color Intensity Invariant Low Level Feature Optimization Framework for Image Quality Assessment.pdf
[2016]Brain-Inspired Deep Networks for Image Aesthetics Assessment.pdf
[2016]Hierarchical aesthetic quality assessment using deep convolutional neural networks.pdf
[2016]ILGNet--Deep Image Aesthetics Classification using Inception Modules and Fine-tuning Connected Layer.pdf
[2016]Image Aesthetic Assessment- An Experimental Survey.pdf
[2016]Visual Aesthetic Quality Assessment with Multi-task Deep Learning.pdf
[2016][CVPR]Composition-preserving Deep Photo Aesthetics Assessment.pdf
[2016][CVPR]Rethinking the Inception Architecture for Computer Vision.pdf
[2016][ECCV]photo aesthetics ranking network with attributes and content adaptation.pdf
[2017]Aesthetic-Driven Image Enhancement by Adversarial Learning.pdf
[2017]NIMA- Neural Image Assessment.pdf
[2017]Quantitative Analysis of Automatic Image Cropping Algorithms- A Dataset and Comparative Study.pdf
[2017][CVPR]A-Lamp- Adaptive Layout-Aware Multi-Patch Deep Convolutional Neural Network for Photo Aesthetic Assessment.pdf
[2017][cvpr]Deep Image Harmonization.pdf
[2017][iccv]Deep aesthetic quality assessment with semantic information.pdf
[2017][iccv]Personalized Image Aesthetics.pdf
[2018]Attention-based Multi-Patch Aggregation for Image Aesthetic Assessment.pdf
```
- 方法众多，但是好的能用的数据集很少， 能用的也就AVA数据集和AADB数据集

# 检测模糊区域
```
[2004]Blur Detection for Digital Images Using Wavelet Transform.pdf
[2014][cvpr]Discriminative Blur Detection Features
```
- point spread function PSF
- 应用
  - 模糊增强
  - 模糊分割
  - 去模糊 deblurring
# 图像质量评估

# TODO
- CS4476.szeliski
  - 应用 DeepNav, IM2GPS, Image retrieval, Image classification, Geolocation, Image Synthesis， OCR, Face Detection, 笑脸检测， 视觉生物，不用密码登录，对象识别， 特效：运动捕捉，特效：捕获形状，体育，医疗图像，智能汽车，交互式游戏，工业机器人，太空视觉，增强现实，虚拟现实， ;领域 安全，健康，保安，舒适，乐趣， 登录；
  - 投影几何 Projective Geometry
    - 消失点和消失线：平行线
      - 世界坐标到图像坐标
      - 直线还是直线
      - 角度和长度丢失
    - 针孔相机模型和相机投影矩阵
    - 齐次坐标
      - 齐次坐标：统一成一个通用公式。 x=K[R t]X
    - 径向扭曲 Radial Distortion
  - Image Formation
    - 
  - 图片压缩，DCT量化, run length encoding， Huffman coding。
  - 人类视觉
    - 人类视觉先过滤各种各样的定向及scale of frequency
    - 中到高频主导视觉
    - 远看图像时，人类会有效子采样图像
  - Image Filtering
    - FFT
    - Filtering in frequency domain
    - Filtering in spacial domain
    - 图片绝大部分是光滑的
    - 采样前先低通滤波
