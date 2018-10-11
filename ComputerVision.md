# Stanford University CS231: 
- 视觉历史

|1959|猫大脑视觉区的电子信号|Simple Cell, Complex Cell, Hypercomplex Cell, Stinulus|
|:--|:--|:--|
|1963|Block World|计算机显示模型，微分，选择的特征|
|1966|The Summer Vision Project|MIT, AI group, pattern recognition|
|1970s|Stages of Visual Representation|input image->edges, zero crossings blobs, bars, ends, virtual lines, groups, curves, boundaries, local surface orientation, discontinuities in depth and in surface orientation, 3D model representation|
|1979|Generialized Cylinder||
|1973|Pictorial Structure||
|1987|Sobel Operator or ...||
|1997|Normalized Cut||
|1999|SIFT&Objedct Recognition||
|2001|Face Detection||
|2005|Histogram of Gradient(HOG)||
|2006|Spatial Pyramid Matching||
|2009|Deformable Part Model||
|2006-2012|PASCAL Visual Object Challenge (20 object categories)|aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor|
|2009|ImageNet|22K categories and 14M images|
|2010-2014|LSVRC-Large Scale Visual Recognition Challenge|1000 categories 1430000 images|

- 图像分类image classification任务
   - google 图片识别
   - 菜品识别 
   - 酒店分类 hotel classification
   - 商品识别 
   - 对象检测 object detection
   - 给图片配字幕 image captioning
   - 动作分类 action classification
   - object detection
   - reasoning

- CNN卷积神经网络for对象识别

![](https://github.com/liangjin2007/data_liangjin/blob/master/semantic%20segmentation.jpg?raw=true)
![](https://github.com/liangjin2007/data_liangjin/blob/master/layout.jpg?raw=true)
![](https://github.com/liangjin2007/data_liangjin/blob/master/3d-relation.jpg?raw=true)
![](https://github.com/liangjin2007/data_liangjin/blob/master/image-captioning.jpg?raw=true)
![](https://github.com/liangjin2007/data_liangjin/blob/master/surface-attribute.jpg?raw=true)
![](https://github.com/liangjin2007/data_liangjin/blob/master/vision-technology.jpg?raw=true)
![](https://github.com/liangjin2007/data_liangjin/blob/master/hard-problem.jpg?raw=true)

- Course Philosophy课程哲学
   - Thorough透彻 and Detailed 详细
      - write from scratch, debug and train CNN
   - Practical实际
      - GPU， scale, distributed optimization, state of the art software tool e.g. Caffe, Torch and Tensorflow
   - State of the art最新
      - Past 1-3 years
   - Fun欢乐
      - Fun topics e.g. Image Captionning(RNN)
      - DeepDream, NeuralStyle, etc

- Grading policy打分规则
   - 3 Problem Sets: 15% x 3 = 45% 三个问题集共占45%
   - Midterm Exam: 15% 其中考试占15%
   - Final Course Project: 40%最后课程项目占40%
      - Milestone: 5% 列出里程碑占5%
      - Final write-up: 35% 最后写出占35%
      - bonus points for exceptional poster presentation不同寻常的展示有加分
   - Late Policy迟到策略
      - 7 free late days - use them in your ways
      - Afterwards, 25% off per day late
      - Not accepted after 3 late days per PS
      - Does not apply to Final Course Project
   - Collaboration Policy
      - Read the student code book, understand what is collaboration and what is academic infraction
      
- 图像分类pipline
   - Python Numpy Tutorial
   - 任务：给定标注集合，给图片打上标注
   - 图像表示： 3D, range[0,255], Channel
   - 挑战：viewpoint change, illumination, deformation, occlusion, background clutter, intraclass variation,
   - 数据驱动方法
      - 收集图像和标注
      - 用机器学习训练一个模型
      - 预测位置的图像测试集
   - 第一个分类器
      - Nearest Neighbor Classifier
      - 相似性定义： distance metric
      - train部分只要记住训练集就可以
   - 近似最近邻 ANN
      - ANN
      - FLANN
   - KNN
      - 最高准确率是多少？
      - 距离用哪种最好？
      - 最好的k是多少？
      - 以上就是KNN的超参数
      - 超参数跟具体问题有关，必须通过尝试才能知道
      - 直接用测试集来改进训练的超参数是不对的，测试集用来衡量预测的泛化效果。
      - 需要从训练集分化出验证集
   - Linear Classifier
      - 参数化方法f(x, W) : x - > score [float]x10
      - 输入是[32x32x3]
      - 输出是score [10]
      - 线性分类器 f(x,W)=Wx+b, W is 10x3072, b is 10x1
         - 像素被转成1列
         - 
# Stanford University CS231a: Computer Vision, From 3D Reconstruction to Recognition
# Stanford University CS231b: The Cutting Edge of Computer Vision
# Stanford University CS231n: CNN for Visual Recognition
# Stanford University CS331: Advanced Reading in Computer Vision
# Stanford University CS431: High-Level Vision




