# 图像特征
- SIFT,HOG,Haar...
- CNN
- Visual Word
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
  
# 分类
- SVM
- CNN
- Softmax
- L-Softmax


# 物体检测
- [RCNN](https://web.njit.edu/~usman/courses/cs698_spring18/RCNN.pdf)
  - ![两步法](https://github.com/liangjin2007/data_liangjin/blob/master/RCNN.jpg?raw=true)
  - selective search
    - segment image into 1k or 2k regions
    - similarity computation
    - merge similar regions
  - similarity
    - color similarity
      - 颜色直方图
    - texture similarity
      - 高斯梯度直方图
  - 

# 相机和图像
  - 针孔相机模型, 数字相机, 图像生成, 数字图像, 线性滤波(去噪，提取信息等), 高斯滤波，correlation and convolution
# 边检测
  - 法向不连续，深度不连续，颜色不连续，照明不连续, 用卷积实现导数，偏导数, 有限差分滤波, 图像梯度, 卷积的导数理论, Derivative of Gaussians, Laplace of Gaussians, 定位边 Canny's Edge Detector, 非极大抑制NMS, 训练边检测，Seam Carving(动态规划法求最小梯度能量路径)
# 图像金字塔Image Pyramids
- 图像下采样， 走样问题（采样频率不够时出现），需要了解图像的结构，极小采样率Nyquist rate； 通过傅立叶变换找到信号的最高频率，设置Nyquist rate为此频率的两倍；高频率由尖锐的边导致，为了1/2下采样能更好的工作，需要降低原图像的最高频率，解决方法：Gaussian滤波来模糊图像，所以图像金子塔的构建过程是这样的，blur->subsampling->blur->subsampling->blur->subsampling...称为Gaussian Pyramids，图形学里称为mip map；

# 图像特征
- 图像局部或者全局的描述。不同的任务要选择不同的特征;提取哪里的特征？提取什么特征？特征如何匹配？；地标检测：不同的位置称为keypoints,不同的特征称为descriptors;跟踪tracking（场景或者人物在往哪里移动）：有特点的关键点及有特点的特征;视频中完全不同的帧的检测：全局特征
，descriptors可以简单，比如只用颜色；角色后的场景是什么类型；图片中两个杯子类型是否一样：每个杯子一个patch,一个patch一个descriptor, descriptor可以比较简单；在图片中找某个patch:每个位置匹配一下，比较像素值，仿射变换不变性；Detection（keypoints or densely）检测，Description描述符(空间范围及内容)，匹配；关键点keypoint检测,应用于图像stitching，[x,y,[x1,x2,...,xn]]，keypoint不能太少也不能太多，结构不清的patch不能检测，带有高对比度改变（梯度）容易检测，直线段不好检测，有两个方向的梯度最容易检测(corners)；如何比较两个patch, WSSD(weighted summed square difference), 数学
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
