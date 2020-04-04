# 数学、算法、库

## 数学
```
\begin{aligned}
Lagrange \ Function: L(t,x(t), x'(t))) \\
Simple \ Functional:A[f] = \int L(t,x(t),x'(t)) dt \\
Euler-Lagrange \ Equation: -\frac{d}{dt}\frac{\partial L}{\partial x'}+\frac{L}{x} = 0 \\
The \ Solution \ to \ minA[f] \ is \ the \ solution \ of \ E-L \ Equation. 
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;Lagrange&space;\&space;Function:&space;L(t,x(t),&space;x'(t)))&space;\\&space;Simple&space;\&space;Functional:A[f]&space;=&space;\int&space;L(t,x(t),x'(t))&space;dt&space;\\&space;Euler-Lagrange&space;\&space;Equation:&space;-\frac{d}{dt}\frac{\partial&space;L}{\partial&space;x'}&plus;\frac{L}{x}&space;=&space;0&space;\\&space;The&space;\&space;Solution&space;\&space;to&space;\&space;minA[f]&space;\&space;is&space;\&space;the&space;\&space;solution&space;\&space;of&space;\&space;E-L&space;\&space;Equation.&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;Lagrange&space;\&space;Function:&space;L(t,x(t),&space;x'(t)))&space;\\&space;Simple&space;\&space;Functional:A[f]&space;=&space;\int&space;L(t,x(t),x'(t))&space;dt&space;\\&space;Euler-Lagrange&space;\&space;Equation:&space;-\frac{d}{dt}\frac{\partial&space;L}{\partial&space;x'}&plus;\frac{L}{x}&space;=&space;0&space;\\&space;The&space;\&space;Solution&space;\&space;to&space;\&space;minA[f]&space;\&space;is&space;\&space;the&space;\&space;solution&space;\&space;of&space;\&space;E-L&space;\&space;Equation.&space;\end{aligned}" title="\begin{aligned} Lagrange \ Function: L(t,x(t), x'(t))) \\ Simple \ Functional:A[f] = \int L(t,x(t),x'(t)) dt \\ Euler-Lagrange \ Equation: -\frac{d}{dt}\frac{\partial L}{\partial x'}+\frac{L}{x} = 0 \\ The \ Solution \ to \ minA[f] \ is \ the \ solution \ of \ E-L \ Equation. \end{aligned}" /></a>



#### 其他

- Covariance Matrix
- Gram Matrix: AA^T
- Jensen不等式 
  - 如果f是凸函数，X是随机变量，那么E[f(X)] >= f(E[X])
- EM
  - [博客](http://www.cnblogs.com/jerrylead/archive/2011/04/06/2006936.html)
  - EM是一种解决存在隐含变量优化问题的有效方法
  - 目标：独立同分布样本{x(i)}, 每个样本隐含一个类别z, 要使得p(x, z)最大。
- Earth Mover's Distance(EMD)
  - [github](https://github.com/wmayner/pyemd)
- MSE
  - keras
  ```
  return K.mean(K.square(y_pred - y_true), axis=-1)
  ```
- PSNR
  - 10*log10(square(L)/MSE)
   - Structural Similarity (SSIM) Index in Image Space(SSIM)
  - tf.image.ssim
  - [skimage](http://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html)
  - [github](https://github.com/jterrace/pyssim)
  - f(l(x,y)*c(x,y)*s(x,y))

- Luminance
  - mean intensity : l(x) = mean(x) = sum(x)/N

- move to center
  - x-l(x) means project x onto the hyperplane of sum(x) = 0.
  
- Contrast
  - standard deviation(root of squared variance) as contrast. Use unbiased estimation 
  - contrast(x) = sqrt(sum(square(x-l(x)))/(N-1))
  
- normalize
  - (x-l(x))/contrast(x)

- l(x,y) range in 0~1 and avoid numeric overflow
  - (2*l(x)*l(y)+C)/(square(l(x))+square(l(y))+C

- structure comparison
  - correlation(inner product) correlation(x, y) = sum((xi-l(x))*(yi-l(y))/(N-1)
  - s(x,y) = (correlation(x, y)+C)/((contrast(x)*contrast(y))+C)

- spearman ranking
  - 度量预测数组和true数组的排序情况。 这个度量在结合mse的情况下，可以用来度量，预测数组的相对排序情况。
  ```
  def spearman_rank2(y_true, y_pred):
    # tf.argsort(values)
    # tf.gather(values, tf.argsort(values))
    
    n = y_pred.shape[0]
    a = [0.0]*n
    b = [0.0]*n
    for i in range(n):
        a[i] = (i, y_pred[i])

    for i in range(n):
        b[i] = (i, y_true[i])
     
    c = np.array(a, dtype=[('x', int), ('y', float)])
    d = np.array(b, dtype=[('x', int), ('y', float)])

    c.sort(order='y')
    d.sort(order='y')

    for i,v in enumerate(c.tolist()):
        a[v[0]] = i
    for i,v in enumerate(d.tolist()):
        b[v[0]] = i
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)

    return 1.0 - (np.sum(np.square(a-b))*6.0/(n**2*n-n))
  ```


## 库（优化方法）
### Eigen
- https://www.jianshu.com/p/931dff3b1b21
- Eigen Documents http://eigen.tuxfamily.org/dox/index.html
  - http://eigen.tuxfamily.org/dox/group__QuickRefPage.html
  - http://eigen.tuxfamily.org/dox/group__SparseQuickRefPage.html
  - examples
    - eigen/doc/examples
    - eigen/tests
    - eigen/demos
    - Vector3f x = A.lu().solve(b);
    - Vector3f x = A.colPivHouseholderQr().solve(b);
    - choleskey model : Matrix2f x = A.ldlt().solve(b);
```
Eigen::MatrixXd V; // #V x 3

Eigen::MatrixXi F; // #F x 3

Eigen::VectorXd Z; Z = V.col(2); // #V x 1

Eigen::RowVector3d(r, g, b); // 1 x 3

Eigen::Vector3d m = V.colwise().minCoeff(); // avoid to write for loops.
Eigen::Vector3d M = V.colwise().maxCoeff(); 

#include <Eigen/Core>
#include <Eigen/LU>
using namespace std;
using namespace Eigen;
int main()
{
   Matrix3f A;
   Vector3f b;
   A << 1,2,3,  4,5,6,  7,8,10;
   b << 3, 3, 4;
   cout << "Here is the matrix A:" << endl << A << endl;
   cout << "Here is the vector b:" << endl << b << endl;
   Vector3f x = A.lu().solve(b);
   cout << "The solution is:" << endl << x << endl;
}
```

### ceres
头文件不多，结构非常清晰。有c版和c++版，下面介绍c++版。
统一入口ceres.h

#### 概念
-CostFunction
  - NormalPrior
  - CostFunctionToFunctor
  - SizedCostFunction
    - AutoDiffCostFunction
      - cost function定义
        - 用来传入样本相关的数据及变量和残差的尺寸
        ```
        auto cost = new AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(
                new ExponentialResidual(data[2 * i], data[2 * i + 1]))
        ```
      - Functor
        - 用来定义残差需要用到的公式及跟样本相关的数据
        ```
        struct ExponentialResidual {
        ExponentialResidual(double x, double y)
            : x_(x), y_(y) {}

        template <typename T> bool operator()(const T* const m,
                                              const T* const c,
                                              T* residual) const {
          residual[0] = y_ - exp(m[0] * x_ + c[0]);
          return true;
        }
        ```
    - NumericDiffCostFunction
  - DynamicCostFunction
    - 成员函数
      - AddParameterBlock(size)
      - SetNumResiduals(size)
    - DynamicAutoDiffCostFunction
    - DynamicNumericDiffCostFunction
    - DynamicCostFunctionToFunctor
- LossFunction
  - 默认是平方，但平方对outlier效果不好，这边还提供了一些别的loss函数
  ```
  class CERES_EXPORT CauchyLoss : public LossFunction {
   public:
    explicit CauchyLoss(double a) : b_(a * a), c_(1 / b_) { }
    virtual void Evaluate(double, double*) const;

   private:
    // b = a^2.
    const double b_;
    // c = 1 / a^2.
    const double c_;
  };
  ```
- Problem
  - 用来定义问题。问题用残差块来定义，残差块用cost函数， loss函数， 变量。 
  ```
  double m = 0.0;
  double c = 0.0;

  Problem problem;
  for (int i = 0; i < kNumObservations; ++i) {
    problem.AddResidualBlock(
        new AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(
            new ExponentialResidual(data[2 * i], data[2 * i + 1])),
        NULL,  // loss function
        &m, &c // variables need to be optimized.
        );
  }
  ```
  
- Solver
  - Solver::Options
  问题定义好了以后要指定优化需要用到的方法，迭代的次数，等参数，在这个类中指定。
  - Solver::Summary
  问题求好以后出的报告在这个类里。
  - Solver::Solve及全局函数Solve
  全局函数里面其实调用的是Solver::Solve
  ```
  Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << "\n";
  std::cout << "Final   m: " << m << " c: " << c << "\n";
  ```
- Covariance
  - 更复杂的数学问题
  - Covariance::Options

- LocalParameterization
  - IdentityParameterization
  - SubsetParameterization
  - QuaternionParameterization
  - EigenQuaternionParameterization
  - HomogeneousVectorParameterization
  - ProductParameterization
  - AutoDiffLocalParameterization
  
- IterationSummary
- IterationCallback
- NumericDiffOptions
- Others
  - rotation
    - 旋转相关的封装在rotation.h中，包括四元数
  - Jet
  - fpclassify
  - CRSMatrix
  - CubicHermiteSpline
  - types.h
    - LinearSolverType
    - PreconditionerType
    - VisibilityClusteringType
    - SparseLinearAlgebraLibraryType
    - DenseLinearAlgebraLibraryType
    - LoggingType
    - MinimizerType
    - LineSearchDirectionType
    - NonlinearConjugateGradientType
    - LineSearchType
    - TrustRegionStrategyType
    - DoglegType
    - TerminationType
    - CallbackReturnType
    - DumpFormatType
    - DimensionType
    - NumericDiffMethodType
    - LineSearchInterpolationType
    - CovarianceAlgorithmType

### libigl
An Open Source Framework For Geometry Processing Programming.

- 只有头文件
- 已知问题 https://libigl.github.io/#known-issues
- 第一章 基础知识
  - 设计原则
  - 下载源代码
  - 编译
  - 例子工程
  - 网格表示
    - 内存高效，缓存友好，避免指针，拷贝和序列化比较方便。
  - 可视化与交互
    - viewer.data().set_data(D)
    - viewer.data().set_colors(C);
    - viewer.data().set_colormap(...);
    - 绘制点、线、label
    - Viewer菜单
    - Multiple Meshes
      - Viewer::append_mesh()
      - selected_data_index
    - Multiple Viewer
      - Viewer::append_core()
      - Viewer::selected_core_index()
      ```
      viewer.callback_post_resize = [&](igl::opengl::glfw::Viewer &v, int w, int h) {
      v.core( left_view).viewport = Eigen::Vector4f(0, 0, w / 2, h);
      v.core(right_view).viewport = Eigen::Vector4f(w / 2, 0, w - (w / 2), h);
  return true;
};
      ```
  - 基本API
    ```
    igl::readOFF(path, V, F);
    igl::writeOBJ("cube.obj", V, F);
    igl::jet(Z, true, C); // transform Z to C.
    // Read scalar function values from a file, U: #V by 1
    VectorXd U;
    igl::readDMAT(TUTORIAL_SHARED_PATH "/cheburashka-scalar.dmat",U);
    
    // Compute per-face normals
    igl::per_face_normals(V,F,N_faces);
    
    // Compute per-vertex normals
    igl::per_vertex_normals(V,F,N_vertices);
    
    // Compute per-corner normals, |dihedral angle| > 20 degrees --> crease
    igl::per_corner_normals(V,F,20,N_corners);
    
    // Compute integral of Gaussian curvature
    igl::gaussian_curvature(V,F,K);
    
    // Compute mass matrix
    SparseMatrix<double> M,Minv;
    igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_DEFAULT,M);
    
    // Compute invert matrix
    igl::invert_diag(M,Minv);
    // Divide by area to get integral average
    K = (Minv*K).eval();
    
    MatrixXd HN;
    SparseMatrix<double> L,M,Minv;
    igl::cotmatrix(V,F,L);
    igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_VORONOI,M);
    igl::invert_diag(M,Minv);
    HN = -Minv*(L*V);
    H = HN.rowwise().norm(); //up to sign, H是每个定点上的平均曲率
    
    // Compute curvature directions via quadric fitting
    MatrixXd PD1,PD2;
    VectorXd PV1,PV2;
    igl::principal_curvature(V,F,PD1,PD2,PV1,PV2);
    // mean curvature
    H = 0.5*(PV1+PV2);
    
    // Average edge length for sizing
    const double avg = igl::avg_edge_length(V,F);
    
    // Compute gradient operator: #F*3 by #V
    SparseMatrix<double> G;
    igl::grad(V,F,G);
    // Compute gradient of U
    MatrixXd GU = Map<const MatrixXd>((G*U).eval().data(),F.rows(),3);
    // Compute gradient magnitude
    const VectorXd GU_mag = GU.rowwise().norm();

    // Compute Laplace-Beltrami operator: #V by #V
    igl::cotmatrix(V,F,L);

    // Alternative construction of same Laplacian
    SparseMatrix<double> G,K;
    // Gradient/Divergence
    igl::grad(V,F,G);
    // Diagonal per-triangle "mass matrix"
    VectorXd dblA;
    igl::doublearea(V,F,dblA);
    // Place areas along diagonal #dim times
    const auto & T = 1.*(dblA.replicate(3,1)*0.5).asDiagonal();
    // Laplacian K built as discrete divergence of gradient or equivalently
    // discrete Dirichelet energy Hessian
    K = -G.transpose() * T * G;
    cout<<"|K-L|: "<<(K-L).norm()<<endl;
    
    // Smoothing
    
    igl::exact_geodesic(V,F,VS,FS,VT,FT,d);
    
    // matrix and linear algebra
    igl::slice(A,R,C,B);
    igl::sort(X,1,true,Y,I);
    igl::sortrows(X,true,Y,I);
    // Find boundary edges
    MatrixXi E;
    igl::boundary_facets(F,E);
    // Find boundary vertices
    VectorXi b,IA,IC;
    igl::unique(E,b,IA,IC);
    // List of all vertex indices
    VectorXi all,in;
    igl::colon<int>(0,V.rows()-1,all);
    // List of interior indices
    igl::setdiff(all,b,in,IA);

    // Construct and slice up Laplacian
    SparseMatrix<double> L,L_in_in,L_in_b;
    igl::cotmatrix(V,F,L);
    igl::slice(L,in,in,L_in_in);
    igl::slice(L,in,b,L_in_b);

    // Dirichlet boundary conditions from z-coordinate
    VectorXd bc;
    VectorXd Z = V.col(2);
    igl::slice(Z,b,bc);

    // 线性方程求解 Solve PDE
    SimplicialLLT<SparseMatrix<double > > solver(-L_in_in);
    VectorXd Z_in = solver.solve(L_in_b*bc);
    // slice into solution
    igl::slice_into(Z_in,in,Z);

    // 带线性方程约束的二次优化问题 general quadratic programs Alternative, short hand
    igl::min_quad_with_fixed_data<double> mqwf;
    // Linear term is 0
    VectorXd B = VectorXd::Zero(V.rows(),1);
    // Empty constraints
    VectorXd Beq;
    SparseMatrix<double> Aeq;
    // Our cotmatrix is _negative_ definite, so flip sign
    igl::min_quad_with_fixed_precompute((-L).eval(),b,Aeq,true,mqwf);
    igl::min_quad_with_fixed_solve(mqwf,B,bc,Beq,Z);
    
    // 带不等式约束的二次规划问题 Quadratic Programming with inequality
    ```
- 第二章 离散几何量和算子
    - 法向 https://github.com/libigl/libigl/blob/master/tutorial/201_Normals/main.cpp
      - Per-face, Per-vertex, Per-corner。Per-corner效果最好
    - 高斯曲率 https://github.com/libigl/libigl/blob/master/tutorial/202_GaussianCurvature/main.cpp
      - 
    - 平均曲率，主曲率 https://github.com/libigl/libigl/blob/master/tutorial/203_CurvatureDirections/main.cpp
    - 梯度 https://github.com/libigl/libigl/blob/master/tutorial/204_Gradient/main.cpp
      - hat function defined on 1-ring of a vertex.
    - 拉普拉斯相关
      - 欧氏空间中的拉普拉斯算子
        - 梯度的散度
      - Laplace-Beltrami operator
        - 一般化到曲面
        - 离散情况有许多种方式
        - cotangent Laplacian
        - divergence theorem to vertex one-rings.
    - 质量矩阵
      - ∇f≈Gf, G is #Fx3 x #V
      - Laplace_Beltrami_Operator(f) = inverse(M)Lf
      - 内积的曲面积分的离散话integration_S(x.y) = x'My
    - 精确离散测地距离
    - 线性方程求解
    - Dirichlet energy subject
    - bi-Laplace equation
    - Quadratic Programming with inequality constraint                    https://github.com/libigl/libigl/blob/master/tutorial/305_QuadraticProgramming/main.cpp
- 第三章 矩阵和线性代数
  - Matlab风格的函数
  - 拉普拉斯方程

- GUI
  - 可视化曲面
    ```
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F); // 拷贝mesh到viewer
    viewer.data().set_normals(N_faces); //
    viewer.data().set_colors(C); // 可视化标量，#C must be #F or #V.
    viewer.callback_key_down = &key_down; // 键盘鼠标交互
    viewer.selected_data_index; // multiple meshes
    viewer.data().add_points(P, Eigen::RowVector3d(r,g,b)); // 挂件
    viewer.data().add_edges(P1,P2,Eigen::RowVector3d(r,g,b)); // 挂件
    viewer.data().add_label(p,str); // 挂件
    viewer.data().show_lines = false; // 关掉wireframe
    viewer.data().set_vertices(U);
    viewer.data().compute_normals();
    viewer.core.align_camera_center(U,F);
    viewer.launch();
    ```
  - 菜单
    - ImGui
    ```
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    menu.callback_draw_viewer_menu = [&](){}
    ```
    - [menu example](https://github.com/libigl/libigl/blob/master/tutorial/106_ViewerMenu/main.cpp)
    
### dlib

### OpenCV
- MSER检测(最大稳定极值区域)![mser](https://github.com/liangjin2007/data_liangjin/blob/master/opencv_mser.jpg?raw=true)
  - 仿射不变性
  - 区域增长
  - 类似于分水岭算法
  - 多级灰度threshold

- Blob检测 用于几何形状提取和分析
  - 圆度
  - 凸性
  - 可扩展到指定的颜色
  
- Edge Detection
  - 求偏导数
    - Scharr
    - Sobel
  - Canny Detection
    - threshold
  
- dft
  "An Intuitive Explanation of Fourier Theory"
  - 离散傅立叶变换
    - 任何信号（我们的case是可视图像）都可以表示成一系列sinusoids函数的和。
    - 输入信号包含三个部分： 空间频率frequency，振幅magnitude，相位phase
    - 从简单到复杂地去理解dft:
        - 1D Fourier Transform: 如果输入是一个sinusoid, 那么输出是一个单个的peak at point f。
        - 2D Fourier Transform: 输出图像中 横轴为原来图像中沿着x方向的频率， 纵轴为原图中沿着y方向的频率， 亮度为原图中的亮度对比度。
    - DC Term:对应于零频率，代表平均亮度
    - nyquist frequency

- facial_features
  可以指定眼睛等位置

- facedetect
  脸部检测
  
- pyramids
  生成图像的金字塔
  
- convexhull
  算一堆点的凸包
  
- morphology
腐蚀/膨胀/开/闭
  - 开运算：先腐蚀再膨胀，用周围原色填补白色小洞。清楚物体外的小孔洞，闭运算填补物体内的小孔洞。
```
st = cv.getStructuringElement(getattr(cv, str_name), (sz, sz))
res = cv.morphologyEx(img, getattr(cv, oper_name), st, iterations=iters)
```
- mouse_and_match
模版匹配，鼠标交互
```
patch = gray[sel[1]:sel[3],sel[0]:sel[2]]
result = cv.matchTemplate(gray,patch,cv.TM_CCOEFF_NORMED)
result = np.abs(result)**3
_val, result = cv.threshold(result, 0.01, 0, cv.THRESH_TOZERO)
result8 = cv.normalize(result,None,0,255,cv.NORM_MINMAX,cv.CV_8U)
cv.imshow("result", result8)
```
    
- video_threaded
  - 线程池创建
  - 线程池消费和生产
  - 循环怎么写，异步操作

- watershed
```
from common import Sketcher
sketch = Sketcher('img', [self.markers_vis, self.markers], self.get_colors)
sketch.show()
sketch.dirty = True 

m = self.markers.copy()
cv.watershed(self.img, m)
overlay = self.colors[np.maximum(m, 0)]
vis = cv.addWeighted(self.img, 0.5, overlay, 0.5, 0.0, dtype=cv.CV_8UC3)
cv.imshow('watershed', vis)
```

- turing
https://softologyblog.wordpress.com/2011/07/05/multi-scale-turing-patterns/

- floodfill
```
cv.floodFill(flooded, mask, seed_pt, (255, 255, 255), (lo,)*3, (hi,)*3, flags)
cv.circle(flooded, seed_pt, 2, (0, 0, 255), -1)
cv.imshow('floodfill', flooded)
```

- kmeans
  - 生成高斯分布的数据
  - 绘制高斯分布的椭圆形状 
```
def make_gaussians(cluster_n, img_size):
    points = []
    ref_distrs = []
    for _i in xrange(cluster_n):
        mean = (0.1 + 0.8*random.rand(2)) * img_size
        a = (random.rand(2, 2)-0.5)*img_size*0.1
        cov = np.dot(a.T, a) + img_size*0.05*np.eye(2)
        n = 100 + random.randint(900)
        pts = random.multivariate_normal(mean, cov, n)
        points.append( pts )
        ref_distrs.append( (mean, cov) )
    points = np.float32( np.vstack(points) )
    return points, ref_distrs
    
def draw_gaussain(img, mean, cov, color):
    x, y = np.int32(mean)
    w, u, _vt = cv.SVDecomp(cov)
    ang = np.arctan2(u[1, 0], u[0, 0])*(180/np.pi)
    s1, s2 = np.sqrt(w)*3.0
    cv.ellipse(img, (x, y), (s1, s2), ang, 0, 360, color, 1, cv.LINE_AA)

points, _ = make_gaussians(cluster_n, img_size)
        
term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
ret, labels, centers = cv.kmeans(points, cluster_n, None, term_crit, 10, 0)

img = np.zeros((img_size, img_size, 3), np.uint8)
for (x, y), label in zip(np.int32(points), labels.ravel()):
    c = list(map(int, colors[label]))

    cv.circle(img, (x, y), 1, c, -1)

cv.imshow('gaussian mixture', img)
```

- edge
Canny Detection
```
edge = cv.Canny(gray, thrs1, thrs2, apertureSize=5)
vis = img.copy()
vis = np.uint8(vis/2.)
vis[edge != 0] = (0, 255, 0)
```

- letter_recog.py 
这个是训练模型的例子。使用cv.ml中的RTree, KNeaerest, Boost, SVM, MLP，默认是训练Random Trees classifier.
model=cv2.ml.RTrees_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses.astype(int))

- browse.py
这个是图片放大器。
small = cv.pyrDown(small)
cv.getRectSubPix(img, (800,600),(x+0.5, y+0.5))

- tst_scene_render.py
演示cv.fillConvexPoly，非常快的一个接口

- peopledetect.py
```
hog = cv.HOGDescriptor()
hog.setSVMDetector( cv.HOGDescriptor_getDefaultPeopleDetector() )  
found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
```

- hist.py
```
hist_item = cv.calcHist([im],[ch],None,[256],[0,256])
print(hist_item.shape) # (256,1)
cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
hist=np.int32(np.around(hist_item))
pts = np.int32(np.column_stack((bins,hist)))
cv.polylines(h,[pts],False,col)
```

- contours.py
levels设成7能显示所有大小的轮廓
```
contours0, hierarchy = cv.findContours( img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = [cv.approxPolyDP(cnt, 3, True) for cnt in contours0]

def update(levels):
    vis = np.zeros((h, w, 3), np.uint8)
    levels = levels - 3
    cv.drawContours( vis, contours, (-1, 2)[levels <= 0], (128,255,255),
        3, cv.LINE_AA, hierarchy, abs(levels) )
    cv.imshow('contours', vis)
```

- find_obj.py
给定图片，根据图片特征去另一张图片中寻找图片

- fitline.py
DT_L2效果最差
```
func = getattr(cv, cur_func_name)
vx, vy, cx, cy = cv.fitLine(np.float32(points), func, 0, 0.01, 0.01)
cv.line(img, (int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)), (0, 0, 255))
```

- squares.py
```
squares = find_squares(img)
cv.drawContours( img, squares, -1, (0, 255, 0), 3 )

if thrs == 0:
    bin = cv.Canny(gray, 0, 50, apertureSize=5)
    bin = cv.dilate(bin, None)
else:
    _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    cnt_len = cv.arcLength(cnt, True)
    cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
    if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
        cnt = cnt.reshape(-1, 2)
        max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
        if max_cos < 0.1:
            squares.append(cnt)
```

- dft.py
  - 计算dft得到实部和虚部，计算magnitude, cv.log(1+magnitude)将dft图像移到中心，cv.normalize归一化。
  - 实际图像x,y两个方向都表示频率。具体的dft值表示

- calibrate.py？
相机标定：输入棋盘格扭曲图像，计算参数矩阵，undistort扭曲图像。求的是内参？还是外参？
接口：
```
found, corners = cv.findChessboardCorners(img, pattern_size)
cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
cv.drawChessboardCorners(vis, pattern_size, corners, found)
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)
newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
```

- camera_calibration_show_extrinsics.py
用matplotlib ax.plot3D()把矩阵画出来

- digits.py
  - err怎么定义
  - confusion matrix怎么定义 https://www.geeksforgeeks.org/confusion-matrix-machine-learning/

- coherence.py
  - Coherence-Enhancing Shock Filters 用来根据输入图像生成艺术性的图片 03年的文章
- gabor_threads.py
  - 
- houghlines.py
- houghcircles.py
  - hough transformation
  
- opt_flow.py
  - 光流， 用于tracking
- stereo_match.py
  - 对极几何，双view
- distrans.py
  - 距离变换
- texture_flow.py 
  - 输入图像，使用cv.cornerEigenValsAndVecs输出flow.
- logpoloar.py
  - ??
- plane_tracker.py tracker
- plane_ar.py
- lk_homography.py
- lk_track.py tracker
- lappyr.py
  - Laplacian Pyramid construction and merging
- kalman.py tracker
  - kalman filter如何使用
- mosse.py tracker
  - CVPR10 Visual Object Tracking using Adaptive Correlation Filters
- feature_homography.py
- grabcut.py
  - 交互式分割前景
- common.py
- inpaint.py
- deconvolution.py
  - cv.dft
- asift.py
  - 类似于find_obj.py, 用于图像匹配
- gaussian_mix.py
  - gaussian混合模型的例子
- camshift.py tracker
  - mean-shift based tracking

- tutorial code
  - svm

####Mat数学计算
- 多维Mat, 如何创建，如何遍历多维数组。
```
想要创建高维数组，主要利用Mat的 Mat(int ndims, const int* sizes, int type, void* data, const size_t* steps=0)；这一成员函数。
假设要创建四维数组，具体事例如下：
int p = 1;
int q = 2;
int t = 3;
int u = 4;
int sizes[] = {p,q,t,u };
int all = p*q*t*u;
float *d1 = new float[all];
for(int i = 0; i < all; i++)
{
    d1[i] = i * 1.0f;
}

Mat a = Mat(4, sizes, CV_32S, d1);

四维数据的访问为：
int n, c, h, w, id;
for (n = 0; n<p;n++)
{
    for (c=0;c<q;c++)
    {
        for (h=0; h<t; h++)
        {
            for (w=0; w<u; w++)
            {
                id = a.step[0] * n + a.step[1] * c + a.step[2] * h + w * a.step[3];
                //cout << id << endl;
                float *p = (float*)(a.data + id);
                cout << *p << endl;
            }
        }
    }
}
```
- vector<Point2f>及Mat等综合使用
```
// vector to Mat
vector<Point2f> landmarks;
auto m = Mat(landmarks);
  
// Mat与Scalar运算
m = m - 0.5;

// cv的矩阵运算可作用于vector
Scalar s = mean(landmarks);


```
  
### taichi
并行，可微
- tensor
pixels = ti.var(dt=ti.f32, shape=(n * 2, n))
- kernel
ti.kernel
- functions
ti.func
- scope
taichi-scope vs python-scope
- parallel for-loop
- structure for-loop
for i, j in pixels:
  ...
```
It is the loop at the outermost scope that gets parallelized, not the outermost loop.

# Good kernel
@ti.func
def foo():
  for i in x:
    ...

# Bad kernel
@ti.func
def bar(k: ti.i32):
  # The outermost scope is a `if` statement, not the struct-for loop!
  if k > 42:
    for i in x:
      ...
```
-syntax
  - Kernel arguments must be type-hinted. Kernels can have at most 8 parameters, e.g.,


