# 人脸

## 参考资源

- [2017]State of the Art on Monocular 3D Face Reconstruction, Tracking, and Applications

- 最近综述性文章[2020]3D Morphable Face Models - Past, Present and Future及文中提到的链接
  - 大量模型和代码 https://github.com/liangjin2007/curated-list-of-awesome-3D-Morphable-Model-software-and-data
```
1. Introduction


2. Face Capture
	2.1 Shape Acquisition
		2.1.1. Geometric methods
		2.1.2. Photometric methods
		2.1.3. Hybrid metohds
	2.2 Appearance Capture
	2.3 Face part specific methods
	2.4 Dynamic capture
	2.5 Publicly available face datasets
	2.6 Open challenges


3. Modeling
	3.1 Shape models
		3.1.1 Global models
		3.1.2 Local models	
	3.2 Expression models
		3.2.1 Additive models
		3.2.2 Multiplicative models
		3.2.3 Nonlinear models
	3.3 Appearance models
		3.3.1 Linear per-vertex models
		3.3.2 Linear texture-space models
		3.3.3 Nonlinear models
	3.4 Joint shape and appearance models
		AAM
	3.5 Correspondence
		3.5.1 Sparse correspondence computation
		3.5.2 Dense correspondence computation
		3.5.3 Jointly solving for correspondence and statistical models
	3.6 Synthesis of novel model instances
	3.7 Publicly available models
	3.8 Open challenges


4. Image Formation
	4.1 Geometric image formation
		Scaled Orthographic
		Affine
		Perspective
	4.2 Photometric image formation
		Reflectance models: BRDF
		Lighting
		Color transformation
	4.3 Rendering and visibility
	4.4 Open Challenges


5. Analysis-By-Synthesis
	5.1 Input Modalities(形式)
		Multi-View Systems
		Monocular RGBD
		Monocular RGB
	5.2 Energy Functions
		Appearance error
			vertex-wise error
			pixel-wise error
		Feature-based enerties : landmarks, keypoints or fiducial points
		Background Modeling
		Occlusions and Segmentation
	5.3 Optimization
		GD
		SGD
		Pseudo-second-order method: Gauss-Newton or Levenberg-Marquardt, BFGS
		hierarchical optimization
		multiresolution
		Probabilistic methods : Bayesian inference得到一个概率分布
	5.4 Open Challenges
		 shape ambiguity
		 illumination ambiguity
	
	
6. Deep Learning
	6.1 Deep Face Models
	6.2 Deep Face Reconstruction
		6.2.1 Supervised Reconstruction
		6.2.2 Self-Supervised Reconstruction
	6.3 Joint Learning of Model and Reconstruction


7. Applications
	7.1 Face Reconstruction
	7.2 Entertainment
		7.2.1 Controlling 3D Avatars for Games and VR
		7.2.2 Virtual Try-On and  Make-Up
		7.2.3 Face Replacement a.k.a. Face Swap
		7.2.4 Face Reenactment and Visual Dubbing.
	7.3 Medial Applications
	7.4 Forensics
	7.5 Cognitive Science, Neuroscience, and Psychology


8. Perspective
	8.1. Global Challenges
	8.2 Scalability
	8.3 Application
	8.4 Outlook

```

## 基本视觉
- [1989]Motion and structure from two perspective views: algorithms, error analysis, and error estimation
已知图像上的2维坐标，如何算出3d

- [1998][IJCV] Determining the Epipolar Geometry and its Uncertainty : A Review
7点法算fundamental matrix
不知内参也不知外参，只能在像素坐标级别算

- [2004][PAMI] An efficient solution to the five-point relative pose problem
已知内参不知外参，称为Structure from Motion 问题， 5点法求解。

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
- facegood
  - https://m.sohu.com/a/315992246_100006767
  - https://www.jiqizhixin.com/articles/2020-03-16-7
  
## RANSAC

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
      即求解wid, wexp。额外还得到了各个相机的外参Mi。? 
    
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

## EigenFace
```
1.脸部图像投影到一个特征空间（Face Space）。这个空间把已知的人脸之间的变化给最好地编码下来。
2.特征空间即eigenfaces，是整套人脸的特征向量。PCA投影。
3.如何做人脸识别？计算新的人脸在eigenfaces中的投影得到weight pattern。比较weight pattern，可以把这个人脸分类为已知和未知。
4.为什么用PCA? 或者说PCA方法的主要假定是什么？ a)特征空间在图像空间中形成一个cluster b)PCA给出恰当的表示。
5.Eigenface如何计算：
5.1.计算平均face： v
5.2.收集训练集跟平均face间的差：总共M个像素，N是训练图片数， A —— M x N， 协方差矩阵C=AA^T的特征向量。 注意C —— M x M的。一般N < M， 则C最多有N-1个有意义的特征向量。
    有个重要的性质是：L=A^TA, L—— N x N。如果a是L的特征向量，则Aa是C的特征向量。
6.eigenfaces仍然是跟训练图像一样大小的图像，隐约可以看出是人脸的形态。
7.训练图像和新图像都能表示为eigenfaces的线性组合。
```

## AAM理解
- AAM Course on ACCV2010
```
- 输入：{Ii, landmarks_i}
- Build 2d landmark shape model
  - Procrustes算法 ： landmark去掉相似性变换，都统一到同一个空间。首先对n个人脸的landmarks计算mean shape。然后使用迭代算法，把n个人脸align到mean shape, 并不断更新mean shape。
  - procruste算法：
    - align 两组vector<Point2f>。
  - PCA ： 得到mean和n shape特征向量, 记作s0, S， 用参数p可以对形状建模。每个landmark点看作一定活动空间之内的变换。
- Learning appearance model which capture appearance variation for example due to identity and illumination.
  - motion model W, piecewise affine warps, W(x;p)
    - triangulate vertices consistent of landmarks.
- Fitting AAM
```
## RBF
- Analysis the input data
  
## 协方差矩阵和相关性矩阵
- covariance matrix
```
Eigen::MatrixXd centered = mat.rowwise() - Eigen::RowVectorXd(mat.colwise().mean());
Eigen::MatrixXd cov = (centered * centered.adjoint()) / double(centered.rows() - 1);
```
- correlation matrtix
```
Eigen::VectorXd diag = cov.diagonal();
Eigen::MatrixXd inv_sqrt_diag;
inv_sqrt_diag.resize(diag.size(), diag.size());
inv_sqrt_diag.fill(0.0);
for (int i = 0; i < diag.size(); i++) {
inv_sqrt_diag(i, i) = 1.0 / sqrt(diag[i]);
};
Eigen::MatrixXd corr = inv_sqrt_diag * cov * inv_sqrt_diag;
```

## ASM (Active Shape Model)
```
1.概念上来说是EigenFace的扩展。
2.形状用landmark点来表示。
3.Statistical Shape Model
4.平均形状
5.什么样的变化是平凡的
6.一个新形状看起来跟训练数据相近吗
7.ASM是怎么做的：
7.1.把训练数据的landmark点进行对齐。Pose & scale注册
7.2.landmark位置的分布上做PCA
7.2.1.每个形状是一套landmarks
7.2.2.形状的维度为#Landmarks * #SpatialDimension, 维度有点大。
7.2.3.每个特征向量也是一个形状。
7.2.4.主要特征向量描述了几乎所有形状变化。
7.2.5.特征值是被特征向量解释的变化（假定高斯分布）
7.2.6.初始化是有关系的
```
## AAM(Active Appearance Model)
- AAM类似于PCA其实是一种图片encoding方法。它跟PCA不同的是它不是线性的。
- **Combined Appearance Model**
```
1.procrustes分析形状进行对齐，然后算出mean
2.将每个训练的灰度级图像warp到mean
3.采样灰度级图像数据: 类似于光栅化的方法变成texture vector， 然后做一个归一化。 g = (g-miu)/var_squared
4.在灰度级图像数据上做PCA
5.连接形状参数和gray-level参数
6.在连接的向量上再做一次PCA
7.如何实现Active Appearance Model
7.1. x = mean(x) + Qs c, g = mean(g) + Qg c, c是参数
7.2. 图像空间中的形状X 可以认为是以上的x再做一个相似变化(旋转，平移，缩放), X = St(x), 其中t=(sx, sy, tx, ty)。(tx, ty)是平移， sx = s Cos(theta) - 1, sy = s Sin(theta). 这里做了简化
使得St+dt(x) = St(Sdt(x))
7.3. 如何理解没有见到过的图片呢？ 理解为一个优化问题，极小化delta(I) = Ii - Im。 Im来自x, g?
7.4. AAM Training 整合先验知识：将error跟parameter adjustment之间的关系用一个模型记录下来。结果为一个简单的线性模型 delta(c) = A delta(I)
7.4.1. shape-normalized representation warp.
7.4.2. calculate image difference using gray level vectors. delta(g) = gi - gm
7.4.3. 更新线性关系： delta(c) = A delta(g)
7.4.4. 需要一个模型: 能容纳大的error range. 对每个参数最优扰动0.5 * 标准偏差
7.5. AAM Search: 参数更新方程为c' = c - A delta(g)
8.AAM Revised
8.1. Shape : s0, si, s = s0 + p (s1, s2, ..., sn)
8.2. Appearance : A0, Ai, A = A0 + lambda (A1, A2, ..., Am). Ai is defned on s0
8.3. AAM Model instance : A(x) = M(W(x;p)). 称为Forward warping. warping A from s0 to s with Warp(x;p). Warp(x;p) is a pixel.
```
- **inverse compositional algorithm**
```
1.procruste得到s0及normalized shapes {sn}
2.计算S={sn-s0}
3.计算pca得到m_S
4.相似变换 正交化 
	float thresh = (float)1e-6;
	dst.resize(src.rows(), src.cols());

	typedef Eigen::Matrix<float, Eigen::Dynamic, 1> ColVector;

	Eigen::Index k = 0; // storing index
	ColVector v(src.rows(), 1);
	
	for (Eigen::Index i = 0; i < src.cols(); i++)
	{
		// Column to orthogonalize
		v = src.col(i);

		// Subtract projections over previous vectors
		for (Eigen::Index j = 0; j < k; j++)
		{
			auto o = dst.col(j);
			v -= o * (o.transpose() * v);  // remove projection
		}

		// Only keep nonzero vectors
		auto nrm = v.norm();
		if (nrm > thresh)
		{
			dst.col(k) = v / nrm;
			k += 1;
		}
	}

	if (k < dst.cols())
		dst.conservativeResize(Eigen::NoChange, k);

```


