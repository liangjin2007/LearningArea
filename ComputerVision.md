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
   - 线性分类器 Linear Classifier
      - 参数化方法f(x, W) : x - > score [float]x10
      - 输入是[32x32x3]
      - 输出是score [10]
      - 线性分类器 f(x,W)=Wx+b, W is 10x3072, b is 10x1
         - 像素被转成1列
   - SVM分类器 SVM Classifier
   
- 损失函数Loss Function和优化optimization
   - margin loss of SVM, similar to hingle loss
   - 优化方法Optimization
      - 求出恰当的W
   - 权重正则化Weight Regularization
      - L1
      - L2
      - Elastic net(L1+L2)
      - Max Norm Regularization
      - Dropout
   - Softmax分类器 Softmax Classifier (多类别逻辑回归 multinomial Logistic Regression)
      - score function
      - 似然函数Likelihood，求参数使得以样本作为输入的条件下，条件概率最大。求出数据集最符合的概率分布。极大化似然概率。
      - P(Y=k|X=xi) = e^sk/sumj(e^sj) 问题：这个公式为什么成立？
   - [Linear Classification Loss Visualization](http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/)
   - 完整损失函数Full loss
      - ![score,loss,full loss](https://github.com/liangjin2007/data_liangjin/blob/master/loss_and_score.jpg?raw=true)
      
   - 优化optimization
      - Strategy#1: Random Search, 15%
      - Strategy#2: Follow the slope
         - 1维 
         - 多维
            - gradients
         - 如何求梯度
            - 数值方法
            - 分析方法
               - 微积分
            - gradient check
         - 梯度下降法 gradient descent
         - minibatch梯度下降
            - weights_grad = evaluate_gradient(loss, data_batch, weights)
            - weights += step_size*weights_grad # perform parameters update
            - update strategy: momentum, Adagrad, RMSProp, Adam, 
         - Image Features
         - Color Histogram
         - HOG/SIFT Feature
            - Many more: GIST, LBP, Texton, SSIM, ...
         - Bag of Words

- 向后回代和神经网络
   - 微分链式法则
   - activations
   - 神经网络
      - 神经元
         - 全连接层
      - 层数
         - 不包含输入层的数目
   - backpropagation
   - 每个node的backword() API
      - 计算gradient
   - 每个node的forward（）API
      - 计算L
   - 作业
      - Assignment: Writing 2layer Net
      - Stage your forward/backward computation!
   - 激活函数
      - ![激活函数](https://github.com/liangjin2007/data_liangjin/blob/master/activation_functions.jpg?raw=true)
   - [基于ConvnetJS实现的2层神经网络Demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html)
   - 总结
   
- 训练神经网络
   - 迁移学习
      - ![迁移学习](https://github.com/liangjin2007/data_liangjin/blob/master/transfer_learning.jpg?raw=true)
      
   - Caffe Model Zoo
      - https://github.com/BVLC/caffe/wiki/Model-Zoo
   - Minibatch SGD
      - Sample a batch of data
      - Forward prop it through the graph, get loss
      - Backprop to calculate the gradients
      - Update the parameters using gradients and optimization method
   
   - 神经网络Neural Network
      - ![neuron](https://github.com/liangjin2007/data_liangjin/blob/master/neuron.jpg?raw=true)
   
   - Example: torch/nn 
      - [torch/nn](https://github.com/torch/nn)
   
   - 神经网络架构
   
   - 训练网络
      - ![training loop](https://github.com/liangjin2007/data_liangjin/blob/master/training_loop.jpg?raw=true)
      - 一次设置 One time setup
         - 激活函数， 预处理, 权重初始化，正则化，梯度检查
      - 动态训练 training dynamics 
         - babysitting the learning process
         - 参数更新，超参数优化
      - 估计 Estimation
   
   - 激活函数Activations(RELU)
      - ![activations](https://github.com/liangjin2007/data_liangjin/blob/master/activation_functions.jpg?raw=true)
      - 饱和神经元杀掉梯度
      - Sigmoid 输出并不是以零为中心的
      - exp计算比较昂贵
      - tanh 饱和的时候仍然会杀掉梯度
      - Relu 不饱和，非常计算高效，收敛比sigmoid/tanh 快6倍
      - Relu 缺点 死掉的Relu不会再激活，再也不会更新
      - Leaky Relu 不会死掉, 非zero-mean centered
      - PRelu 不会死掉 非zero-mean centered
      - Elu 不会死掉， 接近zero-mean centered, 计算昂贵
      - Maxout 双倍参数数， 不会死掉，不会饱和
      
   - 数据预处理Data Preprocessing(subtract mean)
   - 权重初始化Weight Initialization(use Xavier init)
      - small random numbers
      - [2010]Xavier initialization
         - Can't work with Relu
      - He et al[2015]
  
   - 批正则化Batch Normalization
      - improve gradient flow through the network
      - allow higher learning rate
      - reduce the strong dependence on initialization
      - acts as a form of regularization in a funny way, and slightly reduces the need for dropout
      - at test stage, batch normalization diffs differently
      
   - 查看训练情况Babysitting the Learning process
      - preprocess data
      - select the architecture
      - double check the loss is reasonable
      - start with small regularization and find learning rate that makes the loss go down
      - loss explode
      - loss change little
      
   - 超参数优化Hyperparameter Optimization
      - 交叉验证策略cross-validation strategy
         - coarse to fine
         - a few epochs
      - Random search vs Grid search
      - cross-validation center
      - Visualize the loss curve
      - Visualize the accuracy
         - big gap = overfitting
      - Visualize the weight updates/weight magnitutes
      
   - 参数更新主题Parameter update schemes
      - simple gradient descent scheme x += - learning_rate*dx
      - SGD x += - learning_rate*dx
      - momentum update
         - v = mu * v - learning_rate * dx
         - x += v
      - Nesterov Momentum update
         - v = mu * v - learning_rate * d(x+mu*v)
         - x += v
      - AdaGrad update
         - cache += dx**2
         - x += -learning_rate*dx/(np.sqrt(cache)+1e-7)
      - RMSProp update
         - cache = decay_rate*cache + (1-decay_rate)*dx**2
         - x += -learning_rate*dx/(np.sqrt(cache)+1e-7)
      - Adam update
         - ...
      - SGD + Momentum, Adagrad, RMSProp, Adam
      - 二阶优化方法
         - BGFS
         - L-BGFS
         

# Stanford University CS231a: Computer Vision, From 3D Reconstruction to Recognition
# Stanford University CS231b: The Cutting Edge of Computer Vision
# Stanford University CS231n: CNN for Visual Recognition
# Stanford University CS331: Advanced Reading in Computer Vision
# Stanford University CS431: High-Level Vision




