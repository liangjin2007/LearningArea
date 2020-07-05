# 目录
- 1.几何 
- 2.渲染
- 3.动画
- 4 模拟

## 1.几何 
- CS468-spring-12
```
1.基础(网格， 三角网格，混合网格， 流形网格， C0连续， C1连续， C2连续 Genus, Graph definition, Graph embedding, vertex-degree or vertex valance, Euler-poinCare formula, 欧拉氏性数，非流形，)
2.网格数据结构（half edge，face, set, shared vertex,  face based connectivity, edge based connectivity, adjacency matrix, corner table）
3.Open Mesh
4.曲面重建（深度扫描，注册，隐式曲面重建，逐对刚体注册， 3D数据对齐（ICP），奇异值分解可得到R, Point to Plane Error, 全局注册， spread out error, 基于Graph的方法）
5.曲面重建（隐式曲面重建， level set, 特殊情况， SDF， 法相估计， kNN, PCA, Normal orientation propagation, SDF from point and normal, RBF, Marching Cube, Poisson重建， 迭代方法）
6.微分几何（理解光滑性，重要的方向，弯曲性，修改曲面，数学基础。 曲线，曲面，弧长，长度，曲率，挠率，法向，从法向， Frenet Frame, 法曲率， 主曲率， 欧拉定理，曲面分类(isotropic, anisotropic), 主方向，Gauss-Bonnet定理， 基本形式， Laplace Operator, Laplace Beltrami Operator, 离散微分算子， Mesh上的函数， 函数的梯度， 离散Laplace Beltrami算子 Average Region, 离散法向，离散曲率）
7.光顺(网格质量，Filtering, 傅里叶变换，Laplace smoothing, Spectral analysis, Taubin Smoothing, Smoothing by Laplace Eigen-decomposition, Geometry filtering, Surface fairing能量优化，Thin-plane energy, membrance energy)
8.线性求解器（容易矩阵的求解（对角，上三角，下三角）间接求解器（迭代法： Jacobi, Gauss-Seidel, Conjugate Gradient）, 直接求解器（LU, QR， Cholesky）, Multigrid solvers， 稀疏矩阵的分解是不稀疏的，要通过Reorder去让分解也是稀疏的, Under-determinant系统， Over-determinant系统，Efficient Linear System Solvers for Mesh Processing）
9.简化（Decimation方法（vertex clustering, incremental decimation, resampling, approximation））
10.Progressive Mesh
11.细分
12.参数化（应用（纹理映射，法向贴图，Remeshing， 形状插值， 压缩）， 双射， 低扭曲， 高效计算， f（isometric, conformal, equiareal）, conformal map, comformal parameterization, minimal stretch paramterization, isometric(保持第一基本形式)， conformal（第一基本形式保持比例）, equiareal(第一基本形式的行列式保持不变), isometric<-->conformal+equiareal, 黎曼共形定理， 调和映射（曲面上的Laplace-Beltrami算子=0, 不保角，且不保双射， isometric-->conformal-->harmonic，什么情况下是双射的定理Rado-Kneser-Choquet定理），2d Barycentric drawings, Spring System(mean-value weight, harmonic weight, uniform weight), fixed boundary, non-convex boundary, free-boundary）
13.参数化（自由边界（离散第一基本形式 ： MIPS, Stretch optimization, LSCM(conformal, linear), DCP(conformal, linear)， 角度（ABF, ABF++, LinABF）, edge length(Circle Patterns, CPMS(linear), CETM), Balance area/conformality(ARAP), More...）, J, 第一基本形式I与J的关系 I=J^T J）
14.Remeshing(应用（设计，逆向工程，模拟，可视化）,方法分类（基于参数化的方法和基于曲面的方法）， Quad Remeshing)
15.Deformation（Surface deformation, Space deformation, 方法（laplacian coordinates, edge length + dihedral angles, pyramid coordinates, local frames, ...）, Transform propagation(linear damp scale /shear, log scale damp rotation), MLS deformation, Rigid MLS, Deformation with a cage(Barycentric coordinates, mean-value coordinates, harmonic coordinates)）
16.DEC(Gradient Operator grad f, vector calculus in R^3(div v, curl v, laplace v), 散度定理integrate(omega, div v, dA) = integrate(boundary(omega), dot(v, n), dl)， 格林定理integrate(omega, curl v, dA) = integrate(boundary(omega), dot(v, t), dl))
17.Kinect(应用（Motion sensor, skeleton tracking, facial recognition, voice recognition，手持 3d对象建模， 室内户型生成，Kinect Fusion, 动捕数据生成， 深度图像特征）, RGBD Camera(RGB+Depth)->RGBD features(Sparse features+dense point cloud)->RGBD-ICP(RANSAC+ICP)->Global optimization->Point-Cloud Maps->Surfel Maps, Loop closure detection, Pose-graph optimization, sparse bundle adjustment)
18.Shape segmentation（应用（形状分析任务，产品搜索，建模和编辑，匹配，提取，分类和聚类），元素拟合，比如用平面和圆柱拟合形体，方法（hierarchical mesh decomposition（面之间的距离如何定义，概率）， Randomized Cuts, Random walks， K-means, Core extraction, normalized cut, shape diameter function, superviced segmentation ）

```

# 渲染
### PBR理论
- 知乎 https://zhuanlan.zhihu.com/p/53086060
- ![](https://github.com/liangjin2007/data_liangjin/blob/master/PBR%E7%90%86%E8%AE%BA.jpg?raw=true)

### PBRT架构
### Surface Material(bxdfs)
### Hair, Fur, Feathers
Maya-Geo To Maya Hair 2, GMH2 for short, https://www.youtube.com/watch?v=Rtzfkig9-PY
Hair and water interaction http://www.cs.columbia.edu/cg/liquidhair/, https://github.com/nepluno/libWetHair

### Subsurface scattering， 皮肤建模
Diffusion models

### Volumes
just attenuation
single scattering
multiple scattering
Emission
Nested/overlapping volumes
Volumes with motion blur

### 光模拟算法 Light Simulation Algorithm
Integrators:
单向路径追踪uni-directional path tracing
双向路径追踪bi-directional path tracing
VCM
UPBP
specular rays
diffuse rays
camera rays

### Vertex connection and merging: VCM


###  [[cs348b]Image Synthesis Techniques](http://graphics.stanford.edu/courses/cs348b/)

# 建模
### 程式建模
适合于植物，叶子，地形，分形，建筑，城市，纹理，例子流，人群模拟等
参考：Shape Synthesis from Sketches via Procedural Models and Convolutional Networks

### 形状补充
参考: High-Resolution Shape Completion Using Deep Neural Networks for Global Structure and Local Geometry Inference

### 形状分析
- 介绍
  - 理论工具箱
    - 欧氏几何 Euclidean Geometry
    - 微分几何 Differential Geometry
    - 流形 Manifold
    - 高斯曲率 K
    - 平均曲率 H
    - 第一基本形式 I
    - 第二基本形式 II
    - 形状属性 shape properties
    - 距离 distance
    - 流和向量场 flow and vector field
    - 微分算子 differential operators
    - 黎曼观点
      - 认为只需要角度和距离
      - 蚂蚁的观点
    - 几何物理 geometry mechanic
    - 测度几何 metric geometry
    - 最优传输 optimal transport
    - 微分拓扑
      - Differential/Morse/Persistent topology
  - 计算工具箱
    - 形状的概念
      - 三角形网格 triangle mesh
      - 点云 point cloud
      - 逐对距离矩阵 distance matrix
      - 图 graph
      - 三角形汤 triangle soup
      - 以上几乎都是跟近似，距离，曲率相关
      - 近似光滑曲面 smooth surface
    - 三角网格是否有曲率
    - 组合光滑与离散
      - 离散微分几何 discrete differential geometry
      - 离散化 discretization
    - 离散理论平行与微分几何
      - 结构保留
        - 从连续理论里是对的东西在离散化版本中也对
        - 旋转数 turning number
          - ![旋转数](https://github.com/liangjin2007/data_liangjin/blob/master/turning_number.jpg?raw=true)
      - 收敛
        - 当离散化增进近似对质量
        - 能否把所有连续理论都离散化,no
      - 连续 -> 离散化 -> 离散
    - 数值偏微分方程 Numerical PDE
      - Fast Mean Curvature Flow via Finite-Elements Tracking
    - 光滑优化 Smooth Optimization
      - Bijective parameterization with free boundaries
    - 离散优化 Discrete Optimization
      - Mixed-integer quadrangulation
    - 线性代数 Linear Algebra
      - Consistent shape maps via semidefinite programming
      - Efficient preconditioning of Laplacian matrices for computer graphics
    - The Geometry of Geometry
      - Splines in the space of shells
    - 代数和表示理论
      - SO(3)/~
  - 应用领域 Application areas
    - 图形学 Graphics
      - 转化 transfer
      - 探索模式exploiting patterns
      - 提取 retrieval
      - 编辑 editing
    - 视觉 Vision
      - 识别
      - 导航
      - 重建
      - 分割
    - 医疗图像
      - 分析
      - 分割
      - 注册
    - 制造业
      - 扫描
      - 差品检测
    - 建筑
      - 设计和分析
    - 形状集分析
    - 对应
    - 变形迁移
    - 模拟
    - 模拟
    - 科学可视化
    - 分割
    - 计算机视觉
    - 机器学习
    - 统计
- 数值工具
  - 线性问题
    - 线性算子
    - 线性方程组
    - 求逆，尽量避免求逆
    - Direct
      - Dense
        - 高斯消去
      - Sparse
        - Reordering(SuiteSparse, Eigen)
    - Iterative
      - 正定
        - 共轭梯度
      - 对称
        - MINRES, GMRES
      - Generic
        - LSQR
      
    
    - Direct
    - Direct
  - 优化问题
    - 数学分析
      - 连续函数
      - 梯度
      - Jacobian
      - Hessian
      - 多维泰勒展开
      - 找梯度为0的根
      - Critical point
  - 二次优化问题
  - 最小二乘法法是二次
  - 无限制优化
    - 牛顿，共轭梯度，等
  - 等式限制优化
    - 拉格朗日法
  - 变分问题
    - ？
 
- curves
- surfaces
- curvature
- geodesic distance
- inverse distance problem
  - 降维
    - Manifold learning
    - Dimension Reduction
    - Embedding
    - Multidimensional scaling
  - 问题：已知流型上任意2点间的测地距离，求曲面
  - 测度空间（M,d）
    - d(x,y) >= 0
    - d(x,y) = 0  <-> x = y
    - d(x,y) = d(y,x)
    - 三角不等式
      - d(x,z) <= d(x,y)+d(y,z)
  - Rxn, d(x,y) = ||x-y||


### 程式城市生成
- Wave Function Collapse 波函数塌缩
  - 量子力学体系与外界作用后，波函数突变。术语对应关系:
    - Ax=ax
    - 本征值 <--> 特征值
    - 本征态，本征波函态，本征函数 <--> 特征向量
  - 波函数坍缩可以用来解释为何在单次测量中被测定的物理量的值是确定的，尽管多次测量中每次测量值可能都不同。
  


[TOC]


# 已读论文列表
- [2003]Bilateral Mesh Denoising 
  - 两套权重相乘wc, ws
    - wc：与距离相关的smoothing权重是gaussian filter wc=exp(-x^2/sigma_c^2)
    - ws：与像素相关的特征保留权重或者称为相似性权函数是 ws=exp(-x^2/sigma_s^2)
    - https://github.com/liangjin2007/data_liangjin/blob/master/bilateral_filtering.jpg?raw=true
- [2004]A Remeshing Approach to Multiresolution Modeling
  - laplace operator离散化
  - Euler-Lagrange PDE算子线性离散化
  - multigrid solver
  - direct solver
  - iterative solver 
- [2002]Spanning Tree Seams for Reducing Parameterization Distortion of Triangulated Surfaces
  - Dijkstra shortest path algorithm
  - Minimal Spanning Tree algorithm
  
- [2004]MESH SMOOTHING SCHEMES BASED ON OPTIMAL DELAUNAY TRIANGULATIONS
  - laplace smoothing
  - OPTIMAL DELAUNAY TRIANGULATION Smoother
  - Centroid Voronoi Tesselation Smoother
  - 此论文一堆公式，最终得出的是类似于laplace smoothing一样的算子，唯一的不同是权重。
  
- [2002]Least Squares Conformal Maps for Automatic Texture Atlas Generation
  - 共形映射：把圆映射到圆
  - 如何把问题进行转化
  
- [2002]Surface Simplification Using Quadric Error Metrics
  - 优先队列
  - edge swap
  - edge clapse
