## 编译
```
1. 从github clone代码

2. 安装Visual Studio 2022

3. 使用CMake GUI打开源代码，设置build目录，配置为Visual Studio 2022 x64项目。 点击Configure会提示absl找不到，到Build中把BUILD_DEPS勾上，会自动去下载第三方依赖。点击Generate。 打开Visual Studio项目，设置为Release X64(Debug X64编译不过)。

4. 编译。会报错：
   1. 将报错的三个文件的编码用Notepad++修改为使用utf-8-BOM编码。
   2. 将两个libscip和scip项目的编译选项修改一下。
   编译成功。
```
整体上这个工程的编译有点慢。

## 文档
- 主入口 https://developers.google.com/optimization?hl=zh-cn
- 代码中包含的文档 https://github.com/google/or-tools/tree/stable/ortools/constraint_solver/docs
```
此地址中有TSP， VRP,  VRP的进阶的图，可能更适合直接进到这里来看。
```  
- C++头文件中的注释
```
结合代码来看可能更能理解接口的使用和问题的数学定义。
```
- Constraint Programming  http://kti.mff.cuni.cz/~bartak/constraints/index.html
## [示例](https://developers.google.com/optimization/examples?hl=zh-cn)

## 知识点
[指南](https://developers.google.com/optimization/introduction?hl=zh-cn)中提到:
- 约束规划
- 线性和混合整数规划
```
底下调用的是Glop和SCIP包。
```  
- 车辆路线
- 图表算法(查找图表、最短费用流、最大流量和线性总和分配中的最短路径)

## 入门指南
```
对于每种语言，设置和解决问题的基本步骤都是相同的：

导入所需的库，
声明求解器，
创建变量
定义约束条件，
定义目标函数，
调用求解器，
显示结果。
```


## 问题类型
```
线性优化
如上一部分中所述，线性优化问题是指目标函数和约束条件为变量中的线性表达式。

OR 工具中用于此类问题的主要求解器是线性优化求解器，它实际上是多个用于线性和混合整数优化（包括第三方库）的不同库的封装容器。

详细了解线性优化

混合整数优化
混合整数优化问题是指部分或全部变量必须为整数。例如分配问题，需要将一组工作器分配给一组任务。您可以为每个工作器和任务定义一个变量，如果将指定工作器分配给了给定任务，则该变量的值为 1，否则为 0。在本例中，变量只能采用 0 或 1 的值。

详细了解混合整数优化

限制条件优化
约束优化或约束编程 (CP)，可在大量候选集合中确定可行的解决方案，根据任意约束条件对问题进行建模。CP 基于可行性（找到可行的解决方案）而非优化（找到最佳解决方案），并且侧重于约束条件和变量，而非目标函数。不过，CP 可用于解决优化问题，只需比较所有可行解决方案的目标函数值即可。

详细了解限制条件优化

分配
分配问题涉及将一组代理（例如工作器或机器）分配给一组任务，将每个代理分配给特定任务具有固定的费用。问题在于找到总费用最低的分配。分配问题实际上是网络流问题的特殊情况。

详细了解分配

打包
装箱是指将一组不同大小的对象打包到具有不同容量的容器中的问题。目标是根据容器的容量来打包尽可能多的对象。这种特殊情况是 Knapsack 问题，其中只有一个容器。

详细了解装箱

调度
调度问题涉及分配资源以在特定时间执行一组任务。一个重要的示例是求职招聘问题，即在多台机器上处理多个作业。 每个作业都由一系列任务组成，这些任务必须按给定顺序执行，并且每个任务都必须在特定的机器上处理。问题在于如何分配时间表，以便在尽可能短的时间间隔内完成所有作业。

详细了解时间安排

路由
路线规划问题涉及为车队寻找遍历网络的最佳路线，由有向图定义。什么是优化问题？中描述的将包裹分配给送货卡的问题就是路线问题的一个示例。另一个是旅行推销员问题。

详细了解路由

网络流
许多优化问题都可以由由节点和有向弧线组成的有向图表示。例如，运输问题（其中商品通过铁路网运）可以用图表表示，其中弧线是铁路线，节点是配送中心。

在“最大流”问题中，每条弧形都有可以跨越传输的最大容量。问题是分配要在各个弧形运输的商品量，使运输总量尽可能大。

详细了解网络流
```

## MathOpt
MathOpt 是一个用于对数学优化问题进行建模和解决的库，例如线性编程问题 (LP) 或混合整数编程问题 (MIP)。MathOpt 将建模与求解分隔开来，允许用户通过更改枚举（和构建依赖
项）来选择求解器，从而在求解方法之间切换。

- MathOpt支持如下求解器：
```
GLOP
PDLP
CP-SAT
SCIP
GLPK
Gurobi（需要许可）
HiGHS
```

-示例 
```
#include <iostream>
#include <ostream>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "ortools/base/init_google.h"
#include "ortools/math_opt/cpp/math_opt.h"

// Build the model.
namespace math_opt = ::operations_research::math_opt;
math_opt::Model lp_model("getting_started_lp");
const math_opt::Variable x = lp_model.AddContinuousVariable(-1.0, 1.5, "x");
const math_opt::Variable y = lp_model.AddContinuousVariable(0.0, 1.0, "y");
lp_model.AddLinearConstraint(x + y <= 1.5, "c");
lp_model.Maximize(x + 2 * y);

// Set parameters, e.g. turn on logging.
math_opt::SolveArguments args;
args.parameters.enable_output = true;

// Solve and ensure an optimal solution was found with no errors. 使用Glop求解
const absl::StatusOr<math_opt::SolveResult> result =
    math_opt::Solve(lp_model, math_opt::SolverType::kGlop, args);
CHECK_OK(result.status());
CHECK_OK(result->termination.EnsureIsOptimal());

// Print some information from the result.
std::cout << "MathOpt solve succeeded" << std::endl;
std::cout << "Objective value: " << result->objective_value() << std::endl;
std::cout << "x: " << result->variable_values().at(x) << std::endl;
std::cout << "y: " << result->variable_values().at(y) << std::endl;

```



