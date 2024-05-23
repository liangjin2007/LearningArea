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
- ceres [](https://github.com/ceres-solver/ceres-solver)

1. 解非线性最小二乘问题， 带边界约束。

2. 解无约束优化问题

代码示例：
```
	using ceres::DynamicAutoDiffCostFunction;
	using ceres::CostFunction;
	using ceres::Problem;
	using ceres::Solver;
	using ceres::Solve;

  struct ResidualFuncXXX {
     ResidualFuncXXX(...);

     // Critical member function to define weighted residual value
     template <typename T> bool operator()(T const* const* w, T* residual) const {
       residual[0] = xxx;
       residual[1] = xxx;
       residual[2] = xxx;
 
       return true;
     }

  // const members related to residual functions, e.g. constant coefficients, vectors, matrices etc.
  }



```
















  


  
  
