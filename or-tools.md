or-tools是一个Google开发的专门解优化问题的库。

要使用好这个库的难点在于需要理解要解决的问题。

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

## CP-SAT
```
约束优化(CP) ，用于从大量候选方案中确定可行的解决方案。

CP 基于可行性（找到可行的解决方案）而非优化（找出最佳解决方案），并且侧重于约束和变量，而非目标函数。事实上，CP 问题可能甚至没有目标函数 - 目标是通过为问题添加约束条件，将大量可能的解决方案缩小为更易于管理的子集。
```

- 使用CP-SAT求解器：找可行解。
示例：

我们先来看一个简单的问题示例，其中有：

x、y 和 z 这三个变量，每个变量可取以下值：0、1 或 2。

一项限制条件：x != y

```
返回值：
OPTIMAL	找到了可行的最佳解决方案。
FEASIBLE	找到了可行的解决方案，但不确定它是否是最佳解决方案。
INFEASIBLE	事实证明，这个问题不可行。
MODEL_INVALID	给定的 CpModelProto 未通过验证步骤。您可以通过调用 ValidateCpModel(model_proto) 获取详细错误。
UNKNOWN	模型的状态未知，因为在某事物导致求解器停止之前（例如时间限制、内存限制或用户设置的自定义限制）之前未找到解决方案（或者问题未证明不可行）。

```

- 最优可行解
```
  CpModelBuilder cp_model;

  const Domain domain(0, 2);
  const IntVar x = cp_model.NewIntVar(domain).WithName("x");
  const IntVar y = cp_model.NewIntVar(domain).WithName("y");
  const IntVar z = cp_model.NewIntVar(domain).WithName("z");

  cp_model.AddNotEqual(x, y);

  // Solving part.
  const CpSolverResponse response = Solve(cp_model.Build());

  if (response.status() == CpSolverStatus::OPTIMAL ||
      response.status() == CpSolverStatus::FEASIBLE) {
    // Get the value of x in the solution.
    LOG(INFO) << "x = " << SolutionIntegerValue(response, x);
    LOG(INFO) << "y = " << SolutionIntegerValue(response, y);
    LOG(INFO) << "z = " << SolutionIntegerValue(response, z);
  } else {
    LOG(INFO) << "No solution found.";
  }
```
- 所有可行解
```
  CpModelBuilder cp_model;

  const Domain domain(0, 2);
  const IntVar x = cp_model.NewIntVar(domain).WithName("x");
  const IntVar y = cp_model.NewIntVar(domain).WithName("y");
  const IntVar z = cp_model.NewIntVar(domain).WithName("z");

  cp_model.AddNotEqual(x, y);

  Model model;

  int num_solutions = 0;
  model.Add(NewFeasibleSolutionObserver([&](const CpSolverResponse& r) {
    LOG(INFO) << "Solution " << num_solutions;
    LOG(INFO) << "  x = " << SolutionIntegerValue(r, x);
    LOG(INFO) << "  y = " << SolutionIntegerValue(r, y);
    LOG(INFO) << "  z = " << SolutionIntegerValue(r, z);
    num_solutions++;
  }));

  // Tell the solver to enumerate all solutions.
  SatParameters parameters;
  parameters.set_enumerate_all_solutions(true);
  model.Add(NewSatParameters(parameters));
  const CpSolverResponse response = SolveCpModel(cp_model.Build(), &model);

  LOG(INFO) << "Number of solutions found: " << num_solutions;
```

- 整数规划 https://developers.google.com/optimization/cp/cp_example?hl=zh-cn
```
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>

#include "ortools/base/logging.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/util/sorted_interval_list.h"

namespace operations_research {
namespace sat {

void CpSatExample() {
  CpModelBuilder cp_model;

  int64_t var_upper_bound = std::max({50, 45, 37});
  const Domain domain(0, var_upper_bound);
  const IntVar x = cp_model.NewIntVar(domain).WithName("x");
  const IntVar y = cp_model.NewIntVar(domain).WithName("y");
  const IntVar z = cp_model.NewIntVar(domain).WithName("z");

  cp_model.AddLessOrEqual(2 * x + 7 * y + 3 * z, 50);
  cp_model.AddLessOrEqual(3 * x - 5 * y + 7 * z, 45);
  cp_model.AddLessOrEqual(5 * x + 2 * y - 6 * z, 37);

  cp_model.Maximize(2 * x + 2 * y + 3 * z);

  // Solving part.
  const CpSolverResponse response = Solve(cp_model.Build());

  if (response.status() == CpSolverStatus::OPTIMAL ||
      response.status() == CpSolverStatus::FEASIBLE) {
    // Get the value of x in the solution.
    LOG(INFO) << "Maximum of objective function: "
              << response.objective_value();
    LOG(INFO) << "x = " << SolutionIntegerValue(response, x);
    LOG(INFO) << "y = " << SolutionIntegerValue(response, y);
    LOG(INFO) << "z = " << SolutionIntegerValue(response, z);
  } else {
    LOG(INFO) << "No solution found.";
  }

  // Statistics.
  LOG(INFO) << "Statistics";
  LOG(INFO) << CpSolverResponseStats(response);
}

}  // namespace sat
}  // namespace operations_research

int main() {
  operations_research::sat::CpSatExample();
  return EXIT_SUCCESS;
}
```    
- 员工日程安排
- 求职招聘问题
- N 皇后问题
如何将 N 王后放置在 NxN 棋盘上以免它们两人互相攻击？
任何两个王后都不在同一行、列或对角线上。
```
// OR-Tools solution to the N-queens problem.
#include <stdlib.h>

#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "ortools/base/logging.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/model.h"
#include "ortools/sat/sat_parameters.pb.h"
#include "ortools/util/sorted_interval_list.h"

namespace operations_research {
namespace sat {

void NQueensSat(const int board_size) {
  // Instantiate the solver.
  CpModelBuilder cp_model;

  // There are `board_size` number of variables, one for a queen in each column
  // of the board. The value of each variable is the row that the queen is in.
  std::vector<IntVar> queens;
  queens.reserve(board_size);
  Domain range(0, board_size - 1);
  for (int i = 0; i < board_size; ++i) {
    queens.push_back(
        cp_model.NewIntVar(range).WithName("x" + std::to_string(i)));
  }

  // Define constraints.
  // The following sets the constraint that all queens are in different rows.
  cp_model.AddAllDifferent(queens);

  // No two queens can be on the same diagonal.
  std::vector<LinearExpr> diag_1;
  diag_1.reserve(board_size);
  std::vector<LinearExpr> diag_2;
  diag_2.reserve(board_size);
  for (int i = 0; i < board_size; ++i) {
    diag_1.push_back(queens[i] + i);
    diag_2.push_back(queens[i] - i);
  }
  cp_model.AddAllDifferent(diag_1);
  cp_model.AddAllDifferent(diag_2);

  int num_solutions = 0;
  Model model;
  model.Add(NewFeasibleSolutionObserver([&](const CpSolverResponse& response) {
    LOG(INFO) << "Solution " << num_solutions;
    for (int i = 0; i < board_size; ++i) {
      std::stringstream ss;
      for (int j = 0; j < board_size; ++j) {
        if (SolutionIntegerValue(response, queens[j]) == i) {
          // There is a queen in column j, row i.
          ss << "Q";
        } else {
          ss << "_";
        }
        if (j != board_size - 1) ss << " ";
      }
      LOG(INFO) << ss.str();
    }
    num_solutions++;
  }));

  // Tell the solver to enumerate all solutions.
  SatParameters parameters;
  parameters.set_enumerate_all_solutions(true);
  model.Add(NewSatParameters(parameters));

  const CpSolverResponse response = SolveCpModel(cp_model.Build(), &model);
  LOG(INFO) << "Number of solutions found: " << num_solutions;

  // Statistics.
  LOG(INFO) << "Statistics";
  LOG(INFO) << CpSolverResponseStats(response);
}

}  // namespace sat
}  // namespace operations_research

int main(int argc, char** argv) {
  int board_size = 8;
  if (argc > 1) {
    if (!absl::SimpleAtoi(argv[1], &board_size)) {
      LOG(INFO) << "Cannot parse '" << argv[1]
                << "', using the default value of 8.";
      board_size = 8;
    }
  }
  operations_research::sat::NQueensSat(board_size);
  return EXIT_SUCCESS;
}
```
- 密码谜题


## 线性优化 [Mosek建模实战宝典](https://docs.mosek.com/modeling-cookbook/index.html)
- LP/MIP问题
- MPSolver
```
#include <iostream>
#include <memory>

#include "ortools/linear_solver/linear_solver.h"

std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
if (!solver) {
  LOG(WARNING) << "SCIP solver unavailable.";
  return;
}

const double infinity = solver->infinity();
// x and y are non-negative variables.
MPVariable* const x = solver->MakeNumVar(0.0, infinity, "x");
MPVariable* const y = solver->MakeNumVar(0.0, infinity, "y");
LOG(INFO) << "Number of variables = " << solver->NumVariables();

// x + 2*y <= 14.
MPConstraint* const c0 = solver->MakeRowConstraint(-infinity, 14.0);
c0->SetCoefficient(x, 1);
c0->SetCoefficient(y, 2);

// 3*x - y >= 0.
MPConstraint* const c1 = solver->MakeRowConstraint(0.0, infinity);
c1->SetCoefficient(x, 3);
c1->SetCoefficient(y, -1);

// x - y <= 2.
MPConstraint* const c2 = solver->MakeRowConstraint(-infinity, 2.0);
c2->SetCoefficient(x, 1);
c2->SetCoefficient(y, -1);
LOG(INFO) << "Number of constraints = " << solver->NumConstraints();

// Objective function: 3x + 4y.
MPObjective* const objective = solver->MutableObjective();
objective->SetCoefficient(x, 3);
objective->SetCoefficient(y, 4);
objective->SetMaximization();

const MPSolver::ResultStatus result_status = solver->Solve();
// Check that the problem has an optimal solution.
if (result_status != MPSolver::OPTIMAL) {
  LOG(FATAL) << "The problem does not have an optimal solution!";
}

LOG(INFO) << "Solution:";
LOG(INFO) << "Optimal objective value = " << objective->Value();
LOG(INFO) << x->name() << " = " << x->solution_value();
LOG(INFO) << y->name() << " = " << y->solution_value();
```

- 老虎饮食问题
