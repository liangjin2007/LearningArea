# 渲染
### PBR理论
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
