C++开发中经常会用到第三方库，此文档记录这些年的一些经验：

- 编译成目标平台的库
- 使用过的第三方库 


## 编译成目标平台的库

看第三方库是哪个类型的工程？ 目前接触过的有：

  Visual Studio工程

  CMake 工程

  qt工程

  make 工程

  jam 工程

  只有c++源代码

  Header-Only 库

最容易的是Header-Only库，不需要编译。


## 使用过的第三方库
### 求解器
#### ceres [](https://github.com/ceres-solver/ceres-solver)
```
1. 解非线性最小二乘问题， 带边界约束。

2. 解无约束优化问题
```

代码示例：
```
using ceres::DynamicAutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

Problem problem;

struct ResidualFuncXXX {
ResidualFuncXXX(...);

// Critical member function to define weighted residual value
template <typename T> bool operator()(T const* const* w, T* residual) const {
	// e.g.
	residual[0] = xxx;
	residual[1] = xxx;
	residual[2] = xxx;
	
	return true;
}

// const members related to residual functions, e.g. constant coefficients, vectors, matrices etc.
}

// Let n be variable object count

// one parameter block x
std::vector<double> xs(n);
std::vector<double*> parameter_blocks;
parameter_blocks.push_back(&xs[0]);

// Add residuals to cost
auto cost_function_i = new DynamicAutoDiffCostFunction<ResidualFuncXXX, 4>(new ResidualFuncXXX(...));
cost_function_i->AddParameterBlock(n);
cost_function_i->SetNumResiduals(2);
problem.AddResidualBlock(cost_function_i, NULL, parameter_blocks);  

for (i = 0; i < n; i++) {
	problem.SetParameterLowerBound(parameter_blocks[0], i, lower_bound);
	problem.SetParameterUpperBound(parameter_blocks[0], i, upper_bound);
};

Solver::Options options;
options.linear_solver_type = ceres::ITERATIVE_SCHUR;
options.num_threads = 8;
options.num_linear_solver_threads = 8; // only SPARSE_SCHUR can use this
options.minimizer_progress_to_stdout = true;
options.max_num_iterations = 100;

Solver::Summary summary;
Solve(options, &problem, &summary);
std::cout << summary.BriefReport() << "\n";
```

#### mosek64_9_1 [](https://docs.mosek.com/latest/capi/index.html)
线性规划，二次规划等优化问题都可以用mosek求解。 从SDK中寻找[doc/capi.pdf](https://github.com/liangjin2007/math_related/blob/main/mosek_capi.pdf)，可以看到c语言版本的api 。

libigl中有一个将mosek解二次规划的问题封装成Eigen接口。 

这边我重写过一个版本。


### 视觉
#### opencv
使用过的版本为4.11的gpu版本。

在做深度学习期间，还使用过python版的opencv （pip install opencv-python）

```
// 读图片
cv::Mat img = cv::imread(input_img);

// 转换颜色空间，从bgr 2 hsv
cv::Mat img1;
cv::cvtColor(img, img1, cv::COLOR_BGR2HSV);

...

// 矩阵运算
cv::absdiff(img1, img2, img3);
cv::mean


// 灰度图的blob detection
// use blob detector to detect similar pixel components in images. 
std::vector<cv::KeyPoint> keypoints;

// ref: How to setup Blob parameters ? https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/
cv::SimpleBlobDetector::Params params;
// setup params
...

cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
detector->detect(gray, keypoints);


// indexing
uchar value = gray.at<uchar>(py, px);


// 显示图片
cv::imshow("title", img);
cv::waitKey(0);

// 图形绘制
cv::line
cv:circle
cv::putText
cv::rectangle

// 鼠标回调
cv::setMouseCallback(...);

// 常用数据结构
cv::Scalar, cv::Size, cv::Rect, cv::Point, cv::KeyPoint, cv::Point2f

// utilities
cv::format
```

### 其他
#### Eigen
用得比较多，默认col major。

求解稀疏最小二乘

求解RBF，用于做面补驱动。 

Eigen从某一个版本开始支持在CUDA kernel或者device function中使用Eigen，有个类似于EIGEN_GPU的宏控制。

```
Eigen::Map
Eigen::MatrixXf
Eigen::VectorXd
Eigen::RowVectorXd
Eigen::Vector3d
Eigen::Vector3f
Eigen::Quaternion
Eigen::SparseMatrix
Eigen::Matrix
.block<3, 3>(start_row, start_col)
```




















  


  
  
