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
- **指南** https://developers.google.com/optimization/introduction?hl=zh-cn
- 代码中包含的文档 https://github.com/google/or-tools/tree/stable/ortools/constraint_solver/docs
- C++头文件中的注释
- Constraint Programming  http://kti.mff.cuni.cz/~bartak/constraints/index.html
## [示例](https://developers.google.com/optimization/examples?hl=zh-cn)

## 知识点
[**指南**](https://developers.google.com/optimization/introduction?hl=zh-cn)中提到:
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
- 设置求解器限制
```
  CpModelBuilder cp_model;

  const Domain domain(0, 2);
  const IntVar x = cp_model.NewIntVar(domain).WithName("x");
  const IntVar y = cp_model.NewIntVar(domain).WithName("y");
  const IntVar z = cp_model.NewIntVar(domain).WithName("z");

  cp_model.AddNotEqual(x, y);

  // Solving part.
  Model model;

  // Sets a time limit of 10 seconds.
  SatParameters parameters;
  parameters.set_max_time_in_seconds(10.0);   // 求解限制在10秒内
  model.Add(NewSatParameters(parameters));

  // Solve.
  const CpSolverResponse response = SolveCpModel(cp_model.Build(), &model);
  LOG(INFO) << CpSolverResponseStats(response);

  if (response.status() == CpSolverStatus::OPTIMAL) {
    LOG(INFO) << "  x = " << SolutionIntegerValue(response, x);
    LOG(INFO) << "  y = " << SolutionIntegerValue(response, y);
    LOG(INFO) << "  z = " << SolutionIntegerValue(response, z);
  }
```


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
- PDLP的数学背景 https://developers.google.com/optimization/lp/pdlp_math?hl=zh-cn


## 网络流
图库（Graph library），及求解器

### 最大流
```
您需要将材料从节点 0（来源）传输到节点 4（接收器）。弧线旁边的数字是它们的容量，弧形的容量是在固定的时间段内可以跨越它的最大容量。容量是问题的约束条件。

流是指向每条弧形分配一个非负数（即流量），它满足以下流守恒规则：

注意 ：在每个节点中，除了源节点或接收器之外，通向该节点的所有弧线的总流数等于离开该节点的所有弧线的总流数。
最大流问题是找出整个网络的流量总和尽可能大的流。

以下部分介绍的程序可用于查找从来源 (0) 到接收器 (4) 的最大流。
```
- 代码示例
```
  // Instantiate a SimpleMaxFlow solver.
  SimpleMaxFlow max_flow;

  // Define three parallel arrays: start_nodes, end_nodes, and the capacities
  // between each pair. For instance, the arc from node 0 to node 1 has a
  // capacity of 20.
  std::vector<int64_t> start_nodes = {0, 0, 0, 1, 1, 2, 2, 3, 3};
  std::vector<int64_t> end_nodes = {1, 2, 3, 2, 4, 3, 4, 2, 4};
  std::vector<int64_t> capacities = {20, 30, 10, 40, 30, 10, 20, 5, 20};

  // Add each arc.
  for (int i = 0; i < start_nodes.size(); ++i) {
    max_flow.AddArcWithCapacity(start_nodes[i], end_nodes[i], capacities[i]);
  }

  // Find the maximum flow between node 0 and node 4.
  int status = max_flow.Solve(0, 4);

  if (status == MaxFlow::OPTIMAL) {
    LOG(INFO) << "Max flow: " << max_flow.OptimalFlow();
    LOG(INFO) << "";
    LOG(INFO) << "  Arc    Flow / Capacity";
    for (std::size_t i = 0; i < max_flow.NumArcs(); ++i) {
      LOG(INFO) << max_flow.Tail(i) << " -> " << max_flow.Head(i) << "  "
                << max_flow.Flow(i) << "  / " << max_flow.Capacity(i);
    }
  } else {
    LOG(INFO) << "Solving the max flow problem failed. Solver status: "
              << status;
  }
```
### 最小费用流 最低成本流
```
与最大流问题密切相关的是最低成本（最小成本）流问题，图中的每条弧线都有材料跨越它的单位成本。问题是找到总费用最少的数据流。

最低成本流问题还具有特殊节点（称为供应节点或需求节点），它们类似于最大流问题中的来源和接收器。材料从供应节点传输到需求节点。

在供应节点，向数据流添加了正量，即供应量。例如，供应可以代表该节点上的生产。
在需求节点，从流中取出负的金额（需求）。例如，需求可表示该节点上的消耗。
为方便起见，我们将假设除供给或需求节点之外的所有节点的供应（和需求）都为零。

对于最低成本流问题，我们有以下流守恒规则，它将供应和需求考虑在内：
注意 ：在每个节点中，从节点传出的总流量减去通向该节点的总流量便等于该节点的供给（或需求）。
```
每个节点都提供supply。实际有些可以指定为0.

- 代码
```
// From Bradley, Hax and Maganti, 'Applied Mathematical Programming', figure 8.1
#include <cstdint>
#include <vector>

#include "ortools/graph/min_cost_flow.h"

namespace operations_research {
// MinCostFlow simple interface example.
void SimpleMinCostFlowProgram() {
  // Instantiate a SimpleMinCostFlow solver.
  SimpleMinCostFlow min_cost_flow;

  // Define four parallel arrays: sources, destinations, capacities,
  // and unit costs between each pair. For instance, the arc from node 0
  // to node 1 has a capacity of 15.
  std::vector<int64_t> start_nodes = {0, 0, 1, 1, 1, 2, 2, 3, 4};
  std::vector<int64_t> end_nodes = {1, 2, 2, 3, 4, 3, 4, 4, 2};
  std::vector<int64_t> capacities = {15, 8, 20, 4, 10, 15, 4, 20, 5};
  std::vector<int64_t> unit_costs = {4, 4, 2, 2, 6, 1, 3, 2, 3};

  // Define an array of supplies at each node.
  std::vector<int64_t> supplies = {20, 0, 0, -5, -15};

  // Add each arc.
  for (int i = 0; i < start_nodes.size(); ++i) {
    int arc = min_cost_flow.AddArcWithCapacityAndUnitCost(
        start_nodes[i], end_nodes[i], capacities[i], unit_costs[i]);
    if (arc != i) LOG(FATAL) << "Internal error";
  }

  // Add node supplies.
  for (int i = 0; i < supplies.size(); ++i) {
    min_cost_flow.SetNodeSupply(i, supplies[i]);
  }

  // Find the min cost flow.
  int status = min_cost_flow.Solve();

  if (status == MinCostFlow::OPTIMAL) {
    LOG(INFO) << "Minimum cost flow: " << min_cost_flow.OptimalCost();
    LOG(INFO) << "";
    LOG(INFO) << " Arc   Flow / Capacity  Cost";
    for (std::size_t i = 0; i < min_cost_flow.NumArcs(); ++i) {
      int64_t cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i);
      LOG(INFO) << min_cost_flow.Tail(i) << " -> " << min_cost_flow.Head(i)
                << "  " << min_cost_flow.Flow(i) << "  / "
                << min_cost_flow.Capacity(i) << "       " << cost;
    }
  } else {
    LOG(INFO) << "Solving the min cost flow problem failed. Solver status: "
              << status;
  }
```

## 分配/安排问题(Assignment) https://developers.google.com/optimization/assignment?hl=zh-cn
```
最常见的组合优化问题之一是分配问题。举个例子：假设一组工作器需要执行一组任务，并且对于每个工作器和任务，将工作器分配给任务会产生费用。问题在于，每个工作器最多可分配给一个任务，不能有两个工作器执行相同的任务，同时最大限度地降低总费用。
```
**求解方法有**：MIP求解器， CP-SAT求解器， 线性和赋值求解器，最低费用流求解器

- 最低费用流求解器
```
#include <cstdint>
#include <vector>

#include "ortools/graph/min_cost_flow.h"

namespace operations_research {
// MinCostFlow simple interface example.
void AssignmentMinFlow() {
  // Instantiate a SimpleMinCostFlow solver.
  SimpleMinCostFlow min_cost_flow;

  // Define four parallel arrays: sources, destinations, capacities,
  // and unit costs between each pair. For instance, the arc from node 0
  // to node 1 has a capacity of 15.
  const std::vector<int64_t> start_nodes = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                            3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 8};
  const std::vector<int64_t> end_nodes = {1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8,
                                          5, 6, 7, 8, 5, 6, 7, 8, 9, 9, 9, 9};
  const std::vector<int64_t> capacities = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const std::vector<int64_t> unit_costs = {0,  0,   0,  0,   90,  76, 75, 70,
                                           35, 85,  55, 65,  125, 95, 90, 105,
                                           45, 110, 95, 115, 0,   0,  0,  0};

  const int64_t source = 0;
  const int64_t sink = 9;
  const int64_t tasks = 4;
  // Define an array of supplies at each node.
  const std::vector<int64_t> supplies = {tasks, 0, 0, 0, 0, 0, 0, 0, 0, -tasks};

  // Add each arc.
  for (int i = 0; i < start_nodes.size(); ++i) {
    int arc = min_cost_flow.AddArcWithCapacityAndUnitCost(
        start_nodes[i], end_nodes[i], capacities[i], unit_costs[i]);
    if (arc != i) LOG(FATAL) << "Internal error";
  }

  // Add node supplies.
  for (int i = 0; i < supplies.size(); ++i) {
    min_cost_flow.SetNodeSupply(i, supplies[i]);
  }

  // Find the min cost flow.
  int status = min_cost_flow.Solve();

  if (status == MinCostFlow::OPTIMAL) {
    LOG(INFO) << "Total cost: " << min_cost_flow.OptimalCost();
    LOG(INFO) << "";
    for (std::size_t i = 0; i < min_cost_flow.NumArcs(); ++i) {
      // Can ignore arcs leading out of source or into sink.
      if (min_cost_flow.Tail(i) != source && min_cost_flow.Head(i) != sink) {
        // Arcs in the solution have a flow value of 1. Their start and end
        // nodes give an assignment of worker to task.
        if (min_cost_flow.Flow(i) > 0) {
          LOG(INFO) << "Worker " << min_cost_flow.Tail(i)
                    << " assigned to task " << min_cost_flow.Head(i)
                    << " Cost: " << min_cost_flow.UnitCost(i);
        }
      }
    }
  } else {
    LOG(INFO) << "Solving the min cost flow problem failed.";
    LOG(INFO) << "Solver status: " << status;
  }
```

- MIP 求解器
```
#include <memory>
#include <vector>

#include "ortools/base/logging.h"
#include "ortools/linear_solver/linear_solver.h"

namespace operations_research {
void AssignmentMip() {
  // Data
  const std::vector<std::vector<double>> costs{
      {90, 80, 75, 70},   {35, 85, 55, 65},   {125, 95, 90, 95},
      {45, 110, 95, 115}, {50, 100, 90, 100},
  };
  const int num_workers = costs.size();
  const int num_tasks = costs[0].size();

  // Solver
  // Create the mip solver with the SCIP backend.
  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
  if (!solver) {
    LOG(WARNING) << "SCIP solver unavailable.";
    return;
  }

  // Variables
  // x[i][j] is an array of 0-1 variables, which will be 1
  // if worker i is assigned to task j.
  std::vector<std::vector<const MPVariable*>> x(
      num_workers, std::vector<const MPVariable*>(num_tasks));
  for (int i = 0; i < num_workers; ++i) {
    for (int j = 0; j < num_tasks; ++j) {
      x[i][j] = solver->MakeIntVar(0, 1, "");
    }
  }

  // Constraints
  // Each worker is assigned to at most one task.
  for (int i = 0; i < num_workers; ++i) {
    LinearExpr worker_sum;
    for (int j = 0; j < num_tasks; ++j) {
      worker_sum += x[i][j];
    }
    solver->MakeRowConstraint(worker_sum <= 1.0);
  }
  // Each task is assigned to exactly one worker.
  for (int j = 0; j < num_tasks; ++j) {
    LinearExpr task_sum;
    for (int i = 0; i < num_workers; ++i) {
      task_sum += x[i][j];
    }
    solver->MakeRowConstraint(task_sum == 1.0);
  }

  // Objective.
  MPObjective* const objective = solver->MutableObjective();
  for (int i = 0; i < num_workers; ++i) {
    for (int j = 0; j < num_tasks; ++j) {
      objective->SetCoefficient(x[i][j], costs[i][j]);
    }
  }
  objective->SetMinimization();

  // Solve
  const MPSolver::ResultStatus result_status = solver->Solve();

  // Print solution.
  // Check that the problem has a feasible solution.
  if (result_status != MPSolver::OPTIMAL &&
      result_status != MPSolver::FEASIBLE) {
    LOG(FATAL) << "No solution found.";
  }

  LOG(INFO) << "Total cost = " << objective->Value() << "\n\n";

  for (int i = 0; i < num_workers; ++i) {
    for (int j = 0; j < num_tasks; ++j) {
      // Test if x[i][j] is 0 or 1 (with tolerance for floating point
      // arithmetic).
      if (x[i][j]->solution_value() > 0.5) {
        LOG(INFO) << "Worker " << i << " assigned to task " << j
                  << ".  Cost = " << costs[i][j];
      }
    }
  }
}
}  // namespace operations_research

int main(int argc, char** argv) {
  operations_research::AssignmentMip();
  return EXIT_SUCCESS;
}
```

- CP-SAT求解器
```
#include <stdlib.h>

#include <vector>

#include "ortools/base/logging.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"

namespace operations_research {
namespace sat {

void IntegerProgrammingExample() {
  // Data
  const std::vector<std::vector<int>> costs{
      {90, 80, 75, 70},   {35, 85, 55, 65},   {125, 95, 90, 95},
      {45, 110, 95, 115}, {50, 100, 90, 100},
  };
  const int num_workers = static_cast<int>(costs.size());
  const int num_tasks = static_cast<int>(costs[0].size());

  // Model
  CpModelBuilder cp_model;

  // Variables
  // x[i][j] is an array of Boolean variables. x[i][j] is true
  // if worker i is assigned to task j.
  std::vector<std::vector<BoolVar>> x(num_workers,
                                      std::vector<BoolVar>(num_tasks));
  for (int i = 0; i < num_workers; ++i) {
    for (int j = 0; j < num_tasks; ++j) {
      x[i][j] = cp_model.NewBoolVar();
    }
  }

  // Constraints
  // Each worker is assigned to at most one task.
  for (int i = 0; i < num_workers; ++i) {
    cp_model.AddAtMostOne(x[i]);
  }
  // Each task is assigned to exactly one worker.
  for (int j = 0; j < num_tasks; ++j) {
    std::vector<BoolVar> tasks;
    for (int i = 0; i < num_workers; ++i) {
      tasks.push_back(x[i][j]);
    }
    cp_model.AddExactlyOne(tasks);
  }

  // Objective
  LinearExpr total_cost;
  for (int i = 0; i < num_workers; ++i) {
    for (int j = 0; j < num_tasks; ++j) {
      total_cost += x[i][j] * costs[i][j];
    }
  }
  cp_model.Minimize(total_cost);

  // Solve
  const CpSolverResponse response = Solve(cp_model.Build());

  // Print solution.
  if (response.status() == CpSolverStatus::INFEASIBLE) {
    LOG(FATAL) << "No solution found.";
  }

  LOG(INFO) << "Total cost: " << response.objective_value();
  LOG(INFO);
  for (int i = 0; i < num_workers; ++i) {
    for (int j = 0; j < num_tasks; ++j) {
      if (SolutionBooleanValue(response, x[i][j])) {
        LOG(INFO) << "Task " << i << " assigned to worker " << j
                  << ".  Cost: " << costs[i][j];
      }
    }
  }
}
}  // namespace sat
}  // namespace operations_research

int main(int argc, char** argv) {
  operations_research::sat::IntegerProgrammingExample();
  return EXIT_SUCCESS;
}
```

- Worker带有分组的分配问题
```
Worker如果带分组信息，然后每个分组要求不能处理两个任务

// Solve a simple assignment problem.
#include <stdlib.h>

#include <numeric>
#include <vector>

#include "absl/strings/str_format.h"
#include "ortools/base/logging.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"

namespace operations_research {
namespace sat {

void AssignmentTeamsSat() {
  // Data
  const std::vector<std::vector<int>> costs = {{
      {{90, 76, 75, 70}},
      {{35, 85, 55, 65}},
      {{125, 95, 90, 105}},
      {{45, 110, 95, 115}},
      {{60, 105, 80, 75}},
      {{45, 65, 110, 95}},
  }};
  const int num_workers = static_cast<int>(costs.size());
  std::vector<int> all_workers(num_workers);
  std::iota(all_workers.begin(), all_workers.end(), 0);

  const int num_tasks = static_cast<int>(costs[0].size());
  std::vector<int> all_tasks(num_tasks);
  std::iota(all_tasks.begin(), all_tasks.end(), 0);

  const std::vector<int> team1 = {{0, 2, 4}};
  const std::vector<int> team2 = {{1, 3, 5}};
  // Maximum total of tasks for any team
  const int team_max = 2;

  // Model
  CpModelBuilder cp_model;

  // Variables
  // x[i][j] is an array of Boolean variables. x[i][j] is true
  // if worker i is assigned to task j.
  std::vector<std::vector<BoolVar>> x(num_workers,
                                      std::vector<BoolVar>(num_tasks));
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      x[worker][task] = cp_model.NewBoolVar().WithName(
          absl::StrFormat("x[%d,%d]", worker, task));
    }
  }

  // Constraints
  // Each worker is assigned to at most one task.
  for (int worker : all_workers) {
    cp_model.AddAtMostOne(x[worker]);
  }
  // Each task is assigned to exactly one worker.
  for (int task : all_tasks) {
    std::vector<BoolVar> tasks;
    for (int worker : all_workers) {
      tasks.push_back(x[worker][task]);
    }
    cp_model.AddExactlyOne(tasks);
  }

  // Each team takes at most two tasks.
  LinearExpr team1_tasks;
  for (int worker : team1) {
    for (int task : all_tasks) {
      team1_tasks += x[worker][task];
    }
  }
  cp_model.AddLessOrEqual(team1_tasks, team_max);

  LinearExpr team2_tasks;
  for (int worker : team2) {
    for (int task : all_tasks) {
      team2_tasks += x[worker][task];
    }
  }
  cp_model.AddLessOrEqual(team2_tasks, team_max);

  // Objective
  LinearExpr total_cost;
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      total_cost += x[worker][task] * costs[worker][task];
    }
  }
  cp_model.Minimize(total_cost);

  // Solve
  const CpSolverResponse response = Solve(cp_model.Build());

  // Print solution.
  if (response.status() == CpSolverStatus::INFEASIBLE) {
    LOG(FATAL) << "No solution found.";
  }
  LOG(INFO) << "Total cost: " << response.objective_value();
  LOG(INFO);
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      if (SolutionBooleanValue(response, x[worker][task])) {
        LOG(INFO) << "Worker " << worker << " assigned to task " << task
                  << ".  Cost: " << costs[worker][task];
      }
    }
  }
}
}  // namespace sat
}  // namespace operations_research

int main(int argc, char** argv) {
  operations_research::sat::AssignmentTeamsSat();
  return EXIT_SUCCESS;
}




// Solve a simple assignment problem.
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/strings/str_format.h"
#include "ortools/base/logging.h"
#include "ortools/linear_solver/linear_solver.h"

namespace operations_research {
void AssignmentTeamsMip() {
  // Data
  const std::vector<std::vector<int64_t>> costs = {{
      {{90, 76, 75, 70}},
      {{35, 85, 55, 65}},
      {{125, 95, 90, 105}},
      {{45, 110, 95, 115}},
      {{60, 105, 80, 75}},
      {{45, 65, 110, 95}},
  }};
  const int num_workers = costs.size();
  std::vector<int> all_workers(num_workers);
  std::iota(all_workers.begin(), all_workers.end(), 0);

  const int num_tasks = costs[0].size();
  std::vector<int> all_tasks(num_tasks);
  std::iota(all_tasks.begin(), all_tasks.end(), 0);

  const std::vector<int64_t> team1 = {{0, 2, 4}};
  const std::vector<int64_t> team2 = {{1, 3, 5}};
  // Maximum total of tasks for any team
  const int team_max = 2;

  // Solver
  // Create the mip solver with the SCIP backend.
  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
  if (!solver) {
    LOG(WARNING) << "SCIP solver unavailable.";
    return;
  }

  // Variables
  // x[i][j] is an array of 0-1 variables, which will be 1
  // if worker i is assigned to task j.
  std::vector<std::vector<const MPVariable*>> x(
      num_workers, std::vector<const MPVariable*>(num_tasks));
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      x[worker][task] =
          solver->MakeBoolVar(absl::StrFormat("x[%d,%d]", worker, task));
    }
  }

  // Constraints
  // Each worker is assigned to at most one task.
  for (int worker : all_workers) {
    LinearExpr worker_sum;
    for (int task : all_tasks) {
      worker_sum += x[worker][task];
    }
    solver->MakeRowConstraint(worker_sum <= 1.0);
  }
  // Each task is assigned to exactly one worker.
  for (int task : all_tasks) {
    LinearExpr task_sum;
    for (int worker : all_workers) {
      task_sum += x[worker][task];
    }
    solver->MakeRowConstraint(task_sum == 1.0);
  }

  // Each team takes at most two tasks.
  LinearExpr team1_tasks;
  for (int worker : team1) {
    for (int task : all_tasks) {
      team1_tasks += x[worker][task];
    }
  }
  solver->MakeRowConstraint(team1_tasks <= team_max);

  LinearExpr team2_tasks;
  for (int worker : team2) {
    for (int task : all_tasks) {
      team2_tasks += x[worker][task];
    }
  }
  solver->MakeRowConstraint(team2_tasks <= team_max);

  // Objective.
  MPObjective* const objective = solver->MutableObjective();
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      objective->SetCoefficient(x[worker][task], costs[worker][task]);
    }
  }
  objective->SetMinimization();

  // Solve
  const MPSolver::ResultStatus result_status = solver->Solve();

  // Print solution.
  // Check that the problem has a feasible solution.
  if (result_status != MPSolver::OPTIMAL &&
      result_status != MPSolver::FEASIBLE) {
    LOG(FATAL) << "No solution found.";
  }
  LOG(INFO) << "Total cost = " << objective->Value() << "\n\n";
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      // Test if x[i][j] is 0 or 1 (with tolerance for floating point
      // arithmetic).
      if (x[worker][task]->solution_value() > 0.5) {
        LOG(INFO) << "Worker " << worker << " assigned to task " << task
                  << ".  Cost: " << costs[worker][task];
      }
    }
  }
}
}  // namespace operations_research

int main(int argc, char** argv) {
  operations_research::AssignmentTeamsMip();
  return EXIT_SUCCESS;
}

```

- 任务带有大小的分配问题
```
本部分介绍一个分配问题，其中每个任务都有一个“大小”，表示任务需要多少时间或精力。每个工作器执行的任务的总大小具有固定边界。
// Solve assignment problem.
#include <stdlib.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "absl/strings/str_format.h"
#include "ortools/base/logging.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"

namespace operations_research {
namespace sat {

void AssignmentTaskSizes() {
  // Data
  const std::vector<std::vector<int>> costs = {{
      {{90, 76, 75, 70, 50, 74, 12, 68}},
      {{35, 85, 55, 65, 48, 101, 70, 83}},
      {{125, 95, 90, 105, 59, 120, 36, 73}},
      {{45, 110, 95, 115, 104, 83, 37, 71}},
      {{60, 105, 80, 75, 59, 62, 93, 88}},
      {{45, 65, 110, 95, 47, 31, 81, 34}},
      {{38, 51, 107, 41, 69, 99, 115, 48}},
      {{47, 85, 57, 71, 92, 77, 109, 36}},
      {{39, 63, 97, 49, 118, 56, 92, 61}},
      {{47, 101, 71, 60, 88, 109, 52, 90}},
  }};
  const int num_workers = static_cast<int>(costs.size());
  std::vector<int> all_workers(num_workers);
  std::iota(all_workers.begin(), all_workers.end(), 0);

  const int num_tasks = static_cast<int>(costs[0].size());
  std::vector<int> all_tasks(num_tasks);
  std::iota(all_tasks.begin(), all_tasks.end(), 0);

  const std::vector<int64_t> task_sizes = {{10, 7, 3, 12, 15, 4, 11, 5}};
  // Maximum total of task sizes for any worker
  const int total_size_max = 15;

  // Model
  CpModelBuilder cp_model;

  // Variables
  // x[i][j] is an array of Boolean variables. x[i][j] is true
  // if worker i is assigned to task j.
  std::vector<std::vector<BoolVar>> x(num_workers,
                                      std::vector<BoolVar>(num_tasks));
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      x[worker][task] = cp_model.NewBoolVar().WithName(
          absl::StrFormat("x[%d,%d]", worker, task));
    }
  }

  // Constraints
  // Each worker is assigned to at most one task.
  for (int worker : all_workers) {
    LinearExpr task_sum;
    for (int task : all_tasks) {
      task_sum += x[worker][task] * task_sizes[task];
    }
    cp_model.AddLessOrEqual(task_sum, total_size_max);
  }
  // Each task is assigned to exactly one worker.
  for (int task : all_tasks) {
    std::vector<BoolVar> tasks;
    for (int worker : all_workers) {
      tasks.push_back(x[worker][task]);
    }
    cp_model.AddExactlyOne(tasks);
  }

  // Objective
  LinearExpr total_cost;
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      total_cost += x[worker][task] * costs[worker][task];
    }
  }
  cp_model.Minimize(total_cost);

  // Solve
  const CpSolverResponse response = Solve(cp_model.Build());

  // Print solution.
  if (response.status() == CpSolverStatus::INFEASIBLE) {
    LOG(FATAL) << "No solution found.";
  }
  LOG(INFO) << "Total cost: " << response.objective_value();
  LOG(INFO);
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      if (SolutionBooleanValue(response, x[worker][task])) {
        LOG(INFO) << "Worker " << worker << " assigned to task " << task
                  << ".  Cost: " << costs[worker][task];
      }
    }
  }
}
}  // namespace sat
}  // namespace operations_research

int main(int argc, char** argv) {
  operations_research::sat::AssignmentTaskSizes();
  return EXIT_SUCCESS;
}




// Solve a simple assignment problem.
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/strings/str_format.h"
#include "ortools/base/logging.h"
#include "ortools/linear_solver/linear_solver.h"

namespace operations_research {
void AssignmentTeamsMip() {
  // Data
  const std::vector<std::vector<int64_t>> costs = {{
      {{90, 76, 75, 70, 50, 74, 12, 68}},
      {{35, 85, 55, 65, 48, 101, 70, 83}},
      {{125, 95, 90, 105, 59, 120, 36, 73}},
      {{45, 110, 95, 115, 104, 83, 37, 71}},
      {{60, 105, 80, 75, 59, 62, 93, 88}},
      {{45, 65, 110, 95, 47, 31, 81, 34}},
      {{38, 51, 107, 41, 69, 99, 115, 48}},
      {{47, 85, 57, 71, 92, 77, 109, 36}},
      {{39, 63, 97, 49, 118, 56, 92, 61}},
      {{47, 101, 71, 60, 88, 109, 52, 90}},
  }};
  const int num_workers = costs.size();
  std::vector<int> all_workers(num_workers);
  std::iota(all_workers.begin(), all_workers.end(), 0);

  const int num_tasks = costs[0].size();
  std::vector<int> all_tasks(num_tasks);
  std::iota(all_tasks.begin(), all_tasks.end(), 0);

  const std::vector<int64_t> task_sizes = {{10, 7, 3, 12, 15, 4, 11, 5}};
  // Maximum total of task sizes for any worker
  const int total_size_max = 15;

  // Solver
  // Create the mip solver with the SCIP backend.
  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
  if (!solver) {
    LOG(WARNING) << "SCIP solver unavailable.";
    return;
  }

  // Variables
  // x[i][j] is an array of 0-1 variables, which will be 1
  // if worker i is assigned to task j.
  std::vector<std::vector<const MPVariable*>> x(
      num_workers, std::vector<const MPVariable*>(num_tasks));
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      x[worker][task] =
          solver->MakeBoolVar(absl::StrFormat("x[%d,%d]", worker, task));
    }
  }

  // Constraints
  // Each worker is assigned to at most one task.
  for (int worker : all_workers) {
    LinearExpr worker_sum;
    for (int task : all_tasks) {
      worker_sum += LinearExpr(x[worker][task]) * task_sizes[task];
    }
    solver->MakeRowConstraint(worker_sum <= total_size_max);
  }
  // Each task is assigned to exactly one worker.
  for (int task : all_tasks) {
    LinearExpr task_sum;
    for (int worker : all_workers) {
      task_sum += x[worker][task];
    }
    solver->MakeRowConstraint(task_sum == 1.0);
  }

  // Objective.
  MPObjective* const objective = solver->MutableObjective();
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      objective->SetCoefficient(x[worker][task], costs[worker][task]);
    }
  }
  objective->SetMinimization();

  // Solve
  const MPSolver::ResultStatus result_status = solver->Solve();

  // Print solution.
  // Check that the problem has a feasible solution.
  if (result_status != MPSolver::OPTIMAL &&
      result_status != MPSolver::FEASIBLE) {
    LOG(FATAL) << "No solution found.";
  }
  LOG(INFO) << "Total cost = " << objective->Value() << "\n\n";
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      // Test if x[i][j] is 0 or 1 (with tolerance for floating point
      // arithmetic).
      if (x[worker][task]->solution_value() > 0.5) {
        LOG(INFO) << "Worker " << worker << " assigned to task " << task
                  << ".  Cost: " << costs[worker][task];
      }
    }
  }
}
}  // namespace operations_research

int main(int argc, char** argv) {
  operations_research::AssignmentTeamsMip();
  return EXIT_SUCCESS;
}
```

- 带有允许的群组分配
```
本部分介绍了一个分配问题，即只能将某些允许的工作器组分配给任务。在此示例中，有 12 个工作器，编号为 0 - 11。允许的组是以下几对工作器的组合。


  group1 =  [[2, 3],       # Subgroups of workers 0 - 3
             [1, 3],
             [1, 2],
             [0, 1],
             [0, 2]]


group2 =  [[6, 7],       # Subgroups of workers 4 - 7
             [5, 7],
             [5, 6],
             [4, 5],
             [4, 7]]



group3 =  [[10, 11],     # Subgroups of workers 8 - 11
             [9, 11],
             [9, 10],
             [8, 10],
             [8, 11]]

允许的组可以是三对工作器的任意组合，它们分别来自 group1、group2 和 group3。例如，将 [2, 3]、[6, 7] 和 [10, 11] 组合使用可生成允许的组 [2, 3, 6, 7, 10, 11]。由于这三个集合中的每个元素都包含五个元素，因此允许的组总数为 5 * 5 * 5 = 125。

请注意，如果一组工作器属于任何一个允许的组，则可以解决该问题。换言之，可行集由满足任一约束条件的点组成。这是一个非凸问题的示例。 相比之下，上文所述的 MIP 示例则是一个凸形问题：要使点变得可行，必须满足所有约束条件。

对于诸如此类的非凸问题，CP-SAT 求解器通常比 MIP 求解器更快。以下部分介绍了使用 CP-SAT 求解器和 MIP 求解器的求解时间，并比较了这两种求解器的求解时间。


// Solve assignment problem for given group of workers.
#include <stdlib.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "ortools/base/logging.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"

namespace operations_research {
namespace sat {

void AssignmentGroups() {
  // Data
  const std::vector<std::vector<int>> costs = {{
      {{90, 76, 75, 70, 50, 74}},
      {{35, 85, 55, 65, 48, 101}},
      {{125, 95, 90, 105, 59, 120}},
      {{45, 110, 95, 115, 104, 83}},
      {{60, 105, 80, 75, 59, 62}},
      {{45, 65, 110, 95, 47, 31}},
      {{38, 51, 107, 41, 69, 99}},
      {{47, 85, 57, 71, 92, 77}},
      {{39, 63, 97, 49, 118, 56}},
      {{47, 101, 71, 60, 88, 109}},
      {{17, 39, 103, 64, 61, 92}},
      {{101, 45, 83, 59, 92, 27}},
  }};
  const int num_workers = static_cast<int>(costs.size());
  std::vector<int> all_workers(num_workers);
  std::iota(all_workers.begin(), all_workers.end(), 0);

  const int num_tasks = static_cast<int>(costs[0].size());
  std::vector<int> all_tasks(num_tasks);
  std::iota(all_tasks.begin(), all_tasks.end(), 0);

  // Allowed groups of workers:
  const std::vector<std::vector<int64_t>> group1 = {{
      {{0, 0, 1, 1}},  // Workers 2, 3
      {{0, 1, 0, 1}},  // Workers 1, 3
      {{0, 1, 1, 0}},  // Workers 1, 2
      {{1, 1, 0, 0}},  // Workers 0, 1
      {{1, 0, 1, 0}},  // Workers 0, 2
  }};

  const std::vector<std::vector<int64_t>> group2 = {{
      {{0, 0, 1, 1}},  // Workers 6, 7
      {{0, 1, 0, 1}},  // Workers 5, 7
      {{0, 1, 1, 0}},  // Workers 5, 6
      {{1, 1, 0, 0}},  // Workers 4, 5
      {{1, 0, 0, 1}},  // Workers 4, 7
  }};

  const std::vector<std::vector<int64_t>> group3 = {{
      {{0, 0, 1, 1}},  // Workers 10, 11
      {{0, 1, 0, 1}},  // Workers 9, 11
      {{0, 1, 1, 0}},  // Workers 9, 10
      {{1, 0, 1, 0}},  // Workers 8, 10
      {{1, 0, 0, 1}},  // Workers 8, 11
  }};

  // Model
  CpModelBuilder cp_model;

  // Variables
  // x[i][j] is an array of Boolean variables. x[i][j] is true
  // if worker i is assigned to task j.
  std::vector<std::vector<BoolVar>> x(num_workers,
                                      std::vector<BoolVar>(num_tasks));
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      x[worker][task] = cp_model.NewBoolVar().WithName(
          absl::StrFormat("x[%d,%d]", worker, task));
    }
  }

  // Constraints
  // Each worker is assigned to at most one task.
  for (int worker : all_workers) {
    cp_model.AddAtMostOne(x[worker]);
  }
  // Each task is assigned to exactly one worker.
  for (int task : all_tasks) {
    std::vector<BoolVar> tasks;
    for (int worker : all_workers) {
      tasks.push_back(x[worker][task]);
    }
    cp_model.AddExactlyOne(tasks);
  }

  // Create variables for each worker, indicating whether they work on some
  // task.
  std::vector<IntVar> work(num_workers);
  for (int worker : all_workers) {
    work[worker] = IntVar(
        cp_model.NewBoolVar().WithName(absl::StrFormat("work[%d]", worker)));
  }

  for (int worker : all_workers) {
    LinearExpr task_sum;
    for (int task : all_tasks) {
      task_sum += x[worker][task];
    }
    cp_model.AddEquality(work[worker], task_sum);
  }

  // Define the allowed groups of worders
  auto table1 =
      cp_model.AddAllowedAssignments({work[0], work[1], work[2], work[3]});
  for (const auto& t : group1) {
    table1.AddTuple(t);
  }
  auto table2 =
      cp_model.AddAllowedAssignments({work[4], work[5], work[6], work[7]});
  for (const auto& t : group2) {
    table2.AddTuple(t);
  }
  auto table3 =
      cp_model.AddAllowedAssignments({work[8], work[9], work[10], work[11]});
  for (const auto& t : group3) {
    table3.AddTuple(t);
  }

  // Objective
  LinearExpr total_cost;
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      total_cost += x[worker][task] * costs[worker][task];
    }
  }
  cp_model.Minimize(total_cost);

  // Solve
  const CpSolverResponse response = Solve(cp_model.Build());

  // Print solution.
  if (response.status() == CpSolverStatus::INFEASIBLE) {
    LOG(FATAL) << "No solution found.";
  }
  LOG(INFO) << "Total cost: " << response.objective_value();
  LOG(INFO);
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      if (SolutionBooleanValue(response, x[worker][task])) {
        LOG(INFO) << "Worker " << worker << " assigned to task " << task
                  << ".  Cost: " << costs[worker][task];
      }
    }
  }
}
}  // namespace sat
}  // namespace operations_research

int main(int argc, char** argv) {
  operations_research::sat::AssignmentGroups();
  return EXIT_SUCCESS;
}



// Solve a simple assignment problem.
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "ortools/base/logging.h"
#include "ortools/linear_solver/linear_solver.h"

namespace operations_research {
void AssignmentTeamsMip() {
  // Data
  const std::vector<std::vector<int64_t>> costs = {{
      {{90, 76, 75, 70, 50, 74}},
      {{35, 85, 55, 65, 48, 101}},
      {{125, 95, 90, 105, 59, 120}},
      {{45, 110, 95, 115, 104, 83}},
      {{60, 105, 80, 75, 59, 62}},
      {{45, 65, 110, 95, 47, 31}},
      {{38, 51, 107, 41, 69, 99}},
      {{47, 85, 57, 71, 92, 77}},
      {{39, 63, 97, 49, 118, 56}},
      {{47, 101, 71, 60, 88, 109}},
      {{17, 39, 103, 64, 61, 92}},
      {{101, 45, 83, 59, 92, 27}},
  }};
  const int num_workers = costs.size();
  std::vector<int> all_workers(num_workers);
  std::iota(all_workers.begin(), all_workers.end(), 0);

  const int num_tasks = costs[0].size();
  std::vector<int> all_tasks(num_tasks);
  std::iota(all_tasks.begin(), all_tasks.end(), 0);

  // Allowed groups of workers:
  using WorkerIndex = int;
  using Binome = std::pair<WorkerIndex, WorkerIndex>;
  using AllowedBinomes = std::vector<Binome>;
  const AllowedBinomes group1 = {{
      // group of worker 0-3
      {2, 3},
      {1, 3},
      {1, 2},
      {0, 1},
      {0, 2},
  }};

  const AllowedBinomes group2 = {{
      // group of worker 4-7
      {6, 7},
      {5, 7},
      {5, 6},
      {4, 5},
      {4, 7},
  }};

  const AllowedBinomes group3 = {{
      // group of worker 8-11
      {10, 11},
      {9, 11},
      {9, 10},
      {8, 10},
      {8, 11},
  }};

  // Solver
  // Create the mip solver with the SCIP backend.
  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
  if (!solver) {
    LOG(WARNING) << "SCIP solver unavailable.";
    return;
  }

  // Variables
  // x[i][j] is an array of 0-1 variables, which will be 1
  // if worker i is assigned to task j.
  std::vector<std::vector<const MPVariable*>> x(
      num_workers, std::vector<const MPVariable*>(num_tasks));
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      x[worker][task] =
          solver->MakeBoolVar(absl::StrFormat("x[%d,%d]", worker, task));
    }
  }

  // Constraints
  // Each worker is assigned to at most one task.
  for (int worker : all_workers) {
    LinearExpr worker_sum;
    for (int task : all_tasks) {
      worker_sum += x[worker][task];
    }
    solver->MakeRowConstraint(worker_sum <= 1.0);
  }
  // Each task is assigned to exactly one worker.
  for (int task : all_tasks) {
    LinearExpr task_sum;
    for (int worker : all_workers) {
      task_sum += x[worker][task];
    }
    solver->MakeRowConstraint(task_sum == 1.0);
  }

  // Create variables for each worker, indicating whether they work on some
  // task.
  std::vector<const MPVariable*> work(num_workers);
  for (int worker : all_workers) {
    work[worker] = solver->MakeBoolVar(absl::StrFormat("work[%d]", worker));
  }

  for (int worker : all_workers) {
    LinearExpr task_sum;
    for (int task : all_tasks) {
      task_sum += x[worker][task];
    }
    solver->MakeRowConstraint(work[worker] == task_sum);
  }

  // Group1
  {
    MPConstraint* g1 = solver->MakeRowConstraint(1, 1);
    for (int i = 0; i < group1.size(); ++i) {
      // a*b can be transformed into 0 <= a + b - 2*p <= 1 with p in [0,1]
      // p is true if a AND b, false otherwise
      MPConstraint* tmp = solver->MakeRowConstraint(0, 1);
      tmp->SetCoefficient(work[group1[i].first], 1);
      tmp->SetCoefficient(work[group1[i].second], 1);
      MPVariable* p = solver->MakeBoolVar(absl::StrFormat("g1_p%d", i));
      tmp->SetCoefficient(p, -2);

      g1->SetCoefficient(p, 1);
    }
  }
  // Group2
  {
    MPConstraint* g2 = solver->MakeRowConstraint(1, 1);
    for (int i = 0; i < group2.size(); ++i) {
      // a*b can be transformed into 0 <= a + b - 2*p <= 1 with p in [0,1]
      // p is true if a AND b, false otherwise
      MPConstraint* tmp = solver->MakeRowConstraint(0, 1);
      tmp->SetCoefficient(work[group2[i].first], 1);
      tmp->SetCoefficient(work[group2[i].second], 1);
      MPVariable* p = solver->MakeBoolVar(absl::StrFormat("g2_p%d", i));
      tmp->SetCoefficient(p, -2);

      g2->SetCoefficient(p, 1);
    }
  }
  // Group3
  {
    MPConstraint* g3 = solver->MakeRowConstraint(1, 1);
    for (int i = 0; i < group3.size(); ++i) {
      // a*b can be transformed into 0 <= a + b - 2*p <= 1 with p in [0,1]
      // p is true if a AND b, false otherwise
      MPConstraint* tmp = solver->MakeRowConstraint(0, 1);
      tmp->SetCoefficient(work[group3[i].first], 1);
      tmp->SetCoefficient(work[group3[i].second], 1);
      MPVariable* p = solver->MakeBoolVar(absl::StrFormat("g3_p%d", i));
      tmp->SetCoefficient(p, -2);

      g3->SetCoefficient(p, 1);
    }
  }

  // Objective.
  MPObjective* const objective = solver->MutableObjective();
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      objective->SetCoefficient(x[worker][task], costs[worker][task]);
    }
  }
  objective->SetMinimization();

  // Solve
  const MPSolver::ResultStatus result_status = solver->Solve();

  // Print solution.
  // Check that the problem has a feasible solution.
  if (result_status != MPSolver::OPTIMAL &&
      result_status != MPSolver::FEASIBLE) {
    LOG(FATAL) << "No solution found.";
  }
  LOG(INFO) << "Total cost = " << objective->Value() << "\n\n";
  for (int worker : all_workers) {
    for (int task : all_tasks) {
      // Test if x[i][j] is 0 or 1 (with tolerance for floating point
      // arithmetic).
      if (x[worker][task]->solution_value() > 0.5) {
        LOG(INFO) << "Worker " << worker << " assigned to task " << task
                  << ".  Cost: " << costs[worker][task];
      }
    }
  }
}
}  // namespace operations_research

int main(int argc, char** argv) {
  operations_research::AssignmentTeamsMip();
  return EXIT_SUCCESS;
}
```

- 线性和分配器
```
#include "ortools/graph/assignment.h"

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace operations_research {
// Simple Linear Sum Assignment Problem (LSAP).
void AssignmentLinearSumAssignment() {
  SimpleLinearSumAssignment assignment;

  const int num_workers = 4;
  std::vector<int> all_workers(num_workers);
  std::iota(all_workers.begin(), all_workers.end(), 0);

  const int num_tasks = 4;
  std::vector<int> all_tasks(num_tasks);
  std::iota(all_tasks.begin(), all_tasks.end(), 0);

  const std::vector<std::vector<int>> costs = {{
      {{90, 76, 75, 70}},    // Worker 0
      {{35, 85, 55, 65}},    // Worker 1
      {{125, 95, 90, 105}},  // Worker 2
      {{45, 110, 95, 115}},  // Worker 3
  }};

  for (int w : all_workers) {
    for (int t : all_tasks) {
      if (costs[w][t]) {
        assignment.AddArcWithCost(w, t, costs[w][t]);
      }
    }
  }

  SimpleLinearSumAssignment::Status status = assignment.Solve();

  if (status == SimpleLinearSumAssignment::OPTIMAL) {
    LOG(INFO) << "Total cost: " << assignment.OptimalCost();
    for (int worker : all_workers) {
      LOG(INFO) << "Worker " << std::to_string(worker) << " assigned to task "
                << std::to_string(assignment.RightMate(worker)) << ". Cost: "
                << std::to_string(assignment.AssignmentCost(worker)) << ".";
    }
  } else {
    LOG(INFO) << "Solving the linear assignment problem failed.";
  }
}

}  // namespace operations_research

int main() {
  operations_research::AssignmentLinearSumAssignment();
  return EXIT_SUCCESS;
}
```


## 打包问题(packing)
包问题的目标是找到将一组给定大小的项打包到具有固定packing的容器中的最佳方式。packing典型应用是高效地将箱子加载到货车上。 通常，由于容量限制，无法打包所有商品。在这种情况下，问题是找出总大小不超过最大可容纳在容器中的项的子集。

打包问题有多种类型。其中最重要的两个是**背包问题和装箱问题**。

- 背包问题：

在简单的背包问题中，只有一个容器（背包）。这些项具有价值和尺寸，目标是打包具有最高总价值的部分项。

对于价值等于大小的特殊情况，目标是使打包商品的总大小最大化。

多维背包问题，此类问题是指物品有多个物理数量（例如重量和体积），背包对每个数量都有容量。这里，术语不一定是指高度、长度和宽度的常规空间维度。但是，一些问题可能涉及空间维度，例如，寻找将矩形箱打包到矩形存储箱中的最佳方法。

多个背包问题，此类问题涉及多个背包，目标是使所有背包中已打包的物品的总价值最大化。

请注意，您可能是针对单个背包的多维问题，也可能是只有一个维度的多维背包问题。


- 装箱问题：
最广为人知的打包问题之一是装箱，即装箱时有多个容量相同的容器（称为装箱）。bin-packingbin-packing与多背包问题不同，箱子的数量并非固定的。相反，目标是找到容纳所有项的最小数量的分箱。

下面这个简单的示例说明了多背包问题和装箱问题之间的区别。假设一家公司有货运卡车，每辆卡车的承重能力为 18,000 磅，需要运送 130,000 磅的商品。

多个背包：您有五辆卡车，您想加载其中最重的物品。

装箱：您有 20 辆卡车（足以容纳所有物品），并且希望使用最少的卡车来装运所有物品。

### 背包问题(knapsack问题)
- algorithms/knapsack_solver.h
```
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <sstream>
#include <vector>

#include "ortools/algorithms/knapsack_solver.h"

namespace operations_research {
void RunKnapsackExample() {
  // Instantiate the solver.
  KnapsackSolver solver(
      KnapsackSolver::KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
      "KnapsackExample");

  std::vector<int64_t> values = {
      360, 83, 59,  130, 431, 67, 230, 52,  93,  125, 670, 892, 600,
      38,  48, 147, 78,  256, 63, 17,  120, 164, 432, 35,  92,  110,
      22,  42, 50,  323, 514, 28, 87,  73,  78,  15,  26,  78,  210,
      36,  85, 189, 274, 43,  33, 10,  19,  389, 276, 312};

  std::vector<std::vector<int64_t>> weights = {
      {7,  0,  30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0,  36, 3,  8,  15,
       42, 9,  0,  42, 47, 52, 32, 26, 48, 55, 6,  29, 84, 2,  4,  18, 56,
       7,  29, 93, 44, 71, 3,  86, 66, 31, 65, 0,  79, 20, 65, 52, 13}};

  std::vector<int64_t> capacities = {850};

  solver.Init(values, weights, capacities);
  int64_t computed_value = solver.Solve();

  // Print solution
  std::vector<int> packed_items;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (solver.BestSolutionContains(i)) packed_items.push_back(i);
  }
  std::ostringstream packed_items_ss;
  std::copy(packed_items.begin(), packed_items.end() - 1,
            std::ostream_iterator<int>(packed_items_ss, ", "));
  packed_items_ss << packed_items.back();

  std::vector<int64_t> packed_weights;
  packed_weights.reserve(packed_items.size());
  for (const auto& it : packed_items) {
    packed_weights.push_back(weights[0][it]);
  }
  std::ostringstream packed_weights_ss;
  std::copy(packed_weights.begin(), packed_weights.end() - 1,
            std::ostream_iterator<int>(packed_weights_ss, ", "));
  packed_weights_ss << packed_weights.back();

  int64_t total_weights =
      std::accumulate(packed_weights.begin(), packed_weights.end(), int64_t{0});

  LOG(INFO) << "Total value: " << computed_value;
  LOG(INFO) << "Packed items: {" << packed_items_ss.str() << "}";
  LOG(INFO) << "Total weight: " << total_weights;
  LOG(INFO) << "Packed weights: {" << packed_weights_ss.str() << "}";
}
}  // namespace operations_research

int main(int argc, char** argv) {
  operations_research::RunKnapsackExample();
  return EXIT_SUCCESS;
}
```

### 多背包问题
- MPSolver解法
```
// Solve a multiple knapsack problem using a MIP solver.
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/strings/str_format.h"
#include "ortools/linear_solver/linear_expr.h"
#include "ortools/linear_solver/linear_solver.h"

namespace operations_research {

void MultipleKnapsackMip() {
  const std::vector<int> weights = {
      {48, 30, 42, 36, 36, 48, 42, 42, 36, 24, 30, 30, 42, 36, 36}};
  const std::vector<int> values = {
      {10, 30, 25, 50, 35, 30, 15, 40, 30, 35, 45, 10, 20, 30, 25}};
  const int num_items = weights.size();
  std::vector<int> all_items(num_items);
  std::iota(all_items.begin(), all_items.end(), 0);

  const std::vector<int> bin_capacities = {{100, 100, 100, 100, 100}};
  const int num_bins = bin_capacities.size();
  std::vector<int> all_bins(num_bins);
  std::iota(all_bins.begin(), all_bins.end(), 0);

  // Create the mip solver with the SCIP backend.
  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
  if (!solver) {
    LOG(WARNING) << "SCIP solver unavailable.";
    return;
  }

  // Variables.
  // x[i][b] = 1 if item i is packed in bin b.
  std::vector<std::vector<const MPVariable*>> x(
      num_items, std::vector<const MPVariable*>(num_bins));
  for (int i : all_items) {
    for (int b : all_bins) {
      x[i][b] = solver->MakeBoolVar(absl::StrFormat("x_%d_%d", i, b));
    }
  }

  // Constraints.
  // Each item is assigned to at most one bin.
  for (int i : all_items) {
    LinearExpr sum;
    for (int b : all_bins) {
      sum += x[i][b];
    }
    solver->MakeRowConstraint(sum <= 1.0);
  }
  // The amount packed in each bin cannot exceed its capacity.
  for (int b : all_bins) {
    LinearExpr bin_weight;
    for (int i : all_items) {
      bin_weight += LinearExpr(x[i][b]) * weights[i];
    }
    solver->MakeRowConstraint(bin_weight <= bin_capacities[b]);
  }

  // Objective.
  // Maximize total value of packed items.
  MPObjective* const objective = solver->MutableObjective();
  LinearExpr objective_value;
  for (int i : all_items) {
    for (int b : all_bins) {
      objective_value += LinearExpr(x[i][b]) * values[i];
    }
  }
  objective->MaximizeLinearExpr(objective_value);

  const MPSolver::ResultStatus result_status = solver->Solve();

  if (result_status == MPSolver::OPTIMAL) {
    LOG(INFO) << "Total packed value: " << objective->Value();
    double total_weight = 0.0;
    for (int b : all_bins) {
      LOG(INFO) << "Bin " << b;
      double bin_weight = 0.0;
      double bin_value = 0.0;
      for (int i : all_items) {
        if (x[i][b]->solution_value() > 0) {
          LOG(INFO) << "Item " << i << " weight: " << weights[i]
                    << " value: " << values[i];
          bin_weight += weights[i];
          bin_value += values[i];
        }
      }
      LOG(INFO) << "Packed bin weight: " << bin_weight;
      LOG(INFO) << "Packed bin value: " << bin_value;
      total_weight += bin_weight;
    }
    LOG(INFO) << "Total packed weight: " << total_weight;
  } else {
    LOG(INFO) << "The problem does not have an optimal solution.";
  }
}
}  // namespace operations_research

int main(int argc, char** argv) {
  operations_research::MultipleKnapsackMip();
  return EXIT_SUCCESS;
}
```

- CP-SAP解法
```
// Solves a multiple knapsack problem using the CP-SAT solver.
#include <stdlib.h>

#include <map>
#include <numeric>
#include <tuple>
#include <vector>

#include "absl/strings/str_format.h"
#include "ortools/base/logging.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"

namespace operations_research {
namespace sat {

void MultipleKnapsackSat() {
  const std::vector<int> weights = {
      {48, 30, 42, 36, 36, 48, 42, 42, 36, 24, 30, 30, 42, 36, 36}};
  const std::vector<int> values = {
      {10, 30, 25, 50, 35, 30, 15, 40, 30, 35, 45, 10, 20, 30, 25}};
  const int num_items = static_cast<int>(weights.size());
  std::vector<int> all_items(num_items);
  std::iota(all_items.begin(), all_items.end(), 0);

  const std::vector<int> bin_capacities = {{100, 100, 100, 100, 100}};
  const int num_bins = static_cast<int>(bin_capacities.size());
  std::vector<int> all_bins(num_bins);
  std::iota(all_bins.begin(), all_bins.end(), 0);

  CpModelBuilder cp_model;

  // Variables.
  // x[i, b] = 1 if item i is packed in bin b.
  std::map<std::tuple<int, int>, BoolVar> x;
  for (int i : all_items) {
    for (int b : all_bins) {
      auto key = std::make_tuple(i, b);
      x[key] = cp_model.NewBoolVar().WithName(absl::StrFormat("x_%d_%d", i, b));
    }
  }

  // Constraints.
  // Each item is assigned to at most one bin.
  for (int i : all_items) {
    std::vector<BoolVar> copies;
    for (int b : all_bins) {
      copies.push_back(x[std::make_tuple(i, b)]);
    }
    cp_model.AddAtMostOne(copies);
  }

  // The amount packed in each bin cannot exceed its capacity.
  for (int b : all_bins) {
    LinearExpr bin_weight;
    for (int i : all_items) {
      bin_weight += x[std::make_tuple(i, b)] * weights[i];
    }
    cp_model.AddLessOrEqual(bin_weight, bin_capacities[b]);
  }

  // Objective.
  // Maximize total value of packed items.
  LinearExpr objective;
  for (int i : all_items) {
    for (int b : all_bins) {
      objective += x[std::make_tuple(i, b)] * values[i];
    }
  }
  cp_model.Maximize(objective);

  const CpSolverResponse response = Solve(cp_model.Build());

  if (response.status() == CpSolverStatus::OPTIMAL ||
      response.status() == CpSolverStatus::FEASIBLE) {
    LOG(INFO) << "Total packed value: " << response.objective_value();
    double total_weight = 0.0;
    for (int b : all_bins) {
      LOG(INFO) << "Bin " << b;
      double bin_weight = 0.0;
      double bin_value = 0.0;
      for (int i : all_items) {
        auto key = std::make_tuple(i, b);
        if (SolutionIntegerValue(response, x[key]) > 0) {
          LOG(INFO) << "Item " << i << " weight: " << weights[i]
                    << " value: " << values[i];
          bin_weight += weights[i];
          bin_value += values[i];
        }
      }
      LOG(INFO) << "Packed bin weight: " << bin_weight;
      LOG(INFO) << "Packed bin value: " << bin_value;
      total_weight += bin_weight;
    }
    LOG(INFO) << "Total packed weight: " << total_weight;
  } else {
    LOG(INFO) << "The problem does not have an optimal solution.";
  }

  // Statistics.
  LOG(INFO) << "Statistics";
  LOG(INFO) << CpSolverResponseStats(response);
}
}  // namespace sat
}  // namespace operations_research

int main() {
  operations_research::sat::MultipleKnapsackSat();
  return EXIT_SUCCESS;
}
```

### 装箱问题
```
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>

#include "ortools/linear_solver/linear_expr.h"
#include "ortools/linear_solver/linear_solver.h"

namespace operations_research {
struct DataModel {
  const std::vector<double> weights = {48, 30, 19, 36, 36, 27,
                                       42, 42, 36, 24, 30};
  const int num_items = weights.size();
  const int num_bins = weights.size();
  const int bin_capacity = 100;
};

void BinPackingMip() {
  DataModel data;

  // Create the mip solver with the SCIP backend.
  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
  if (!solver) {
    LOG(WARNING) << "SCIP solver unavailable.";
    return;
  }

  std::vector<std::vector<const MPVariable*>> x(
      data.num_items, std::vector<const MPVariable*>(data.num_bins));
  for (int i = 0; i < data.num_items; ++i) {
    for (int j = 0; j < data.num_bins; ++j) {
      x[i][j] = solver->MakeIntVar(0.0, 1.0, "");
    }
  }
  // y[j] = 1 if bin j is used.
  std::vector<const MPVariable*> y(data.num_bins);
  for (int j = 0; j < data.num_bins; ++j) {
    y[j] = solver->MakeIntVar(0.0, 1.0, "");
  }

  // Create the constraints.
  // Each item is in exactly one bin.
  for (int i = 0; i < data.num_items; ++i) {
    LinearExpr sum;
    for (int j = 0; j < data.num_bins; ++j) {
      sum += x[i][j];
    }
    solver->MakeRowConstraint(sum == 1.0);
  }
  // For each bin that is used, the total packed weight can be at most
  // the bin capacity.
  for (int j = 0; j < data.num_bins; ++j) {
    LinearExpr weight;
    for (int i = 0; i < data.num_items; ++i) {
      weight += data.weights[i] * LinearExpr(x[i][j]);
    }
    solver->MakeRowConstraint(weight <= LinearExpr(y[j]) * data.bin_capacity);
  }

  // Create the objective function.
  MPObjective* const objective = solver->MutableObjective();
  LinearExpr num_bins_used;
  for (int j = 0; j < data.num_bins; ++j) {
    num_bins_used += y[j];
  }
  objective->MinimizeLinearExpr(num_bins_used);

  const MPSolver::ResultStatus result_status = solver->Solve();

  // Check that the problem has an optimal solution.
  if (result_status != MPSolver::OPTIMAL) {
    std::cerr << "The problem does not have an optimal solution!";
    return;
  }
  std::cout << "Number of bins used: " << objective->Value() << std::endl
            << std::endl;
  double total_weight = 0;
  for (int j = 0; j < data.num_bins; ++j) {
    if (y[j]->solution_value() == 1) {
      std::cout << "Bin " << j << std::endl << std::endl;
      double bin_weight = 0;
      for (int i = 0; i < data.num_items; ++i) {
        if (x[i][j]->solution_value() == 1) {
          std::cout << "Item " << i << " - Weight: " << data.weights[i]
                    << std::endl;
          bin_weight += data.weights[i];
        }
      }
      std::cout << "Packed bin weight: " << bin_weight << std::endl
                << std::endl;
      total_weight += bin_weight;
    }
  }
  std::cout << "Total packed weight: " << total_weight << std::endl;
}
}  // namespace operations_research

int main(int argc, char** argv) {
  operations_research::BinPackingMip();
  return EXIT_SUCCESS;
}
```



## 路由Routing
```
最常见的优化任务之一是车辆路线，其目标是为前往一组地点的车队找到最佳路线。通常，“最佳”是指总距离或费用最少的路线。 以下是路由问题的一些示例：

一家包裹配送服务公司想要为司机指定送货路线。
一家有线电视公司希望为技术人员分配用于拨打住宅服务电话的路由。
一家拼车公司想要为司机分配上车和下车路线。
其中最著名的路线问题是旅行推销员问题 (TSP)：对于需要探访不同营业地点的客户并返回出发地的销售人员，找出最短的路线。TSP 可以用图表示，其中节点对应位置，而边（或弧形）表示位置之间的直接行程。例如，下图显示了一个仅有四个位置（分别标记为 A、B、C 和 D）的 TSP。任意两个位置之间的距离由连接这两个位置的边旁边的数字指定。

tsp 动画

通过计算所有可能路线的距离，可以看到最短路线是 ACDBA，总距离为 35 + 30 + 15 + 10 = 90。

地点越多，问题就越严重。上面的示例中只有六条路线。但如果有 10 个位置（不计算起点），则路由数量为 362880。如果是 20 个营业地点，此数量会跳到 2432902008176640000。 详尽搜索所有可能的路线可以保证找到最短路径，但这对于除一小部分位置以外的所有位置来说很难计算。对于更大的问题，需要使用优化技术智能搜索解决方案空间，并找到最佳（或接近优化）的解决方案。

更通用的 TSP 版本是车辆路线问题 (VRP)，其中有多辆车辆。在大多数情况下，VRP 都有限制：例如，车辆可能具有承载最大重量或最大体积的物品容量，或者驾驶员可能需要在客户要求的指定时间范围内造访某些地点。


OR-Tools 可以解决多种类型的 VRP 问题，包括：

旅行销售人员问题，即只有一辆车的典型路线问题。
车辆路线问题，是对多辆车辆的 TSP 的泛化。
具有容量限制的 VRP，即车辆可携带商品的最大容量。
设有时间范围的 VRP：车辆必须按照指定的时间间隔到访地点。
具有资源限制的 VRP，例如在仓库（路线的起点）装卸车辆的空间或人员。
存在访问量下降的 VRP。在这种情况下，车辆并非必须造访所有地点，但必须针对每次访问丢失而支付罚金。

车辆路线问题的限制
车辆路线问题从本质上说是难以解决的：解决问题所需的时间会随着问题规模成倍增长。对于足够大的问题，OR-Tools（或任何其他路由软件）可能需要数年时间才能找到最佳解决方案。因此，OR-Tools 有时会返回良好的解决方案，但并非最佳解决方案。如需找到更好的解决方案，请更改求解器的搜索选项。如需查看示例，请参阅更改搜索策略。

我们还应该补充一点，还有一些其他求解器，例如 Concorde，专用于求解非常大的 TSP 以获得最优性，这些求解器在这方面超过了 OR-Tools。但是，OR-Tools 提供了一个更好的平台，用于解决更宽泛的路由问题，这些问题包含的限制超出纯 TSP 的限制。

```


### TSP问题
```
创建数据
struct DataModel {
  const std::vector<std::vector<int64_t>> distance_matrix{
      {0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972},
      {2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579},
      {713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260},
      {1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987},
      {1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371},
      {1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999},
      {2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701},
      {213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099},
      {2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600},
      {875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162},
      {1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200},
      {2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504},
      {1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0},
  };
  const int num_vehicles = 1;
  const RoutingIndexManager::NodeIndex depot{0};
};   


创建路由模型
DataModel data;
RoutingIndexManager manager(data.distance_matrix.size(), data.num_vehicles,
                            data.depot);
//// Create Routing Index Manager, starts and ends
//RoutingIndexManager manager(data.distance_matrix.size(), data.num_vehicles,
//                           data.starts, data.ends);
RoutingModel routing(manager);


创建距离回调
如需使用路由求解器，您需要创建一个距离（或公交）回调：该函数可接受任意一对位置并返回它们之间的距离。最简单的方法是使用距离矩阵。
const int transit_callback_index = routing.RegisterTransitCallback(
    [&data, &manager](const int64_t from_index,
                      const int64_t to_index) -> int64_t {
      // Convert from routing variable Index to distance matrix NodeIndex.
      const int from_node = manager.IndexToNode(from_index).value();
      const int to_node = manager.IndexToNode(to_index).value();
      return data.distance_matrix[from_node][to_node];
    });

设置旅行花销（Cost）
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index);

在此示例中，弧线费用评估器是 transit_callback_index，它是求解器对距离回调的内部引用。这意味着，任何两个位置之间的行程费用只是它们之间的距离。不过，总体而言，费用还可能会涉及到其他因素。
此外，您还可以使用 routing.SetArcCostEvaluatorOfVehicle() 方法定义多个弧形评估器，以评估车辆在营业地点之间行驶的情况。例如，如果车辆的速度不同，那么您可以将不同地点之间的行程费用定义为距离除以车辆速度（也就是行程时间）。


设置搜索参数
RoutingSearchParameters searchParameters = DefaultRoutingSearchParameters();
searchParameters.set_first_solution_strategy(
    FirstSolutionStrategy::PATH_CHEAPEST_ARC);


求解并打印解决方案
const Assignment* solution = routing.SolveWithParameters(searchParameters);
//! @brief Print the solution.
//! @param[in] manager Index manager used.
//! @param[in] routing Routing solver used.
//! @param[in] solution Solution found by the solver.
void PrintSolution(const RoutingIndexManager& manager,
                   const RoutingModel& routing, const Assignment& solution) {
  // Inspect solution.
  LOG(INFO) << "Objective: " << solution.ObjectiveValue() << " miles";
  int64_t index = routing.Start(0);
  LOG(INFO) << "Route:";
  int64_t distance{0};
  std::stringstream route;
  while (!routing.IsEnd(index)) {
    route << manager.IndexToNode(index).value() << " -> ";
    const int64_t previous_index = index;
    index = solution.Value(routing.NextVar(index));
    distance += routing.GetArcCostForVehicle(previous_index, index, int64_t{0});
  }
  LOG(INFO) << route.str() << manager.IndexToNode(index).value();
  LOG(INFO) << "Route distance: " << distance << "miles";
  LOG(INFO) << "";
  LOG(INFO) << "Advanced usage:";
  LOG(INFO) << "Problem solved in " << routing.solver()->wall_time() << "ms";
};
PrintSolution(manager, routing, *solution);

将路线保存到列表或数组
作为直接输出解决方案的替代方案，您可以将路由（或 VRP 的路由）保存到列表或数组中。这样做的好处是，您可以在日后需要时利用这些路线。例如，您可以使用不同的参数多次运行程序，并将返回的解决方案中的路线保存到文件中进行比较。


std::vector<std::vector<int>> GetRoutes(const Assignment& solution,
                                        const RoutingModel& routing,
                                        const RoutingIndexManager& manager) {
  // Get vehicle routes and store them in a two dimensional array, whose
  // i, j entry is the node for the jth visit of vehicle i.
  std::vector<std::vector<int>> routes(manager.num_vehicles());
  // Get routes.
  for (int vehicle_id = 0; vehicle_id < manager.num_vehicles(); ++vehicle_id) {
    int64_t index = routing.Start(vehicle_id);
    routes[vehicle_id].push_back(manager.IndexToNode(index).value());
    while (!routing.IsEnd(index)) {
      index = solution.Value(routing.NextVar(index));
      routes[vehicle_id].push_back(manager.IndexToNode(index).value());
    }
  }
  return routes;
}

```

- 更改搜索策略
```
RoutingSearchParameters searchParameters = DefaultRoutingSearchParameters();
searchParameters.set_local_search_metaheuristic(
    LocalSearchMetaheuristic::GUIDED_LOCAL_SEARCH);
searchParameters.mutable_time_limit()->set_seconds(30);
search_parameters.set_log_search(true);
```  

### VRP问题
在车辆路线规划问题 (VRP) 中，目标是为访问一组位置的多辆车辆找出最佳路线。（如果只有一辆车，这种情况就会归为销售人员出差问题。）

但是，什么是 VRP 的“最佳路线”呢？其中一个答案是总距离最短的路线。但是，如果没有其他约束条件，则最佳解决方案是仅分配一辆车来访问所有位置，并找到该车辆的最短路线。这本质上与 TSP 的问题相同。

定义最佳路线的更好方法是，尽量缩短所有车辆中最长的单个路线的长度。如果目标是尽快完成所有提交，这就是正确的定义。下面的 VRP 示例查找了以这种方式定义的最佳路线。

在后面的部分中，我们将介绍通过添加车辆约束条件来泛化 TSP 的其他方法，包括：

容量限制：车辆需要在到达的每个地点自提物品，但有最大承载能力。

时间窗口：必须在特定时间范围内访问每个营业地点。


```
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <vector>

#include "ortools/constraint_solver/routing.h"
#include "ortools/constraint_solver/routing_enums.pb.h"
#include "ortools/constraint_solver/routing_index_manager.h"
#include "ortools/constraint_solver/routing_parameters.h"

namespace operations_research {
struct DataModel {
  const std::vector<std::vector<int64_t>> distance_matrix{
      {0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468,
       776, 662},
      {548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674,
       1016, 868, 1210},
      {776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130,
       788, 1552, 754},
      {696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822,
       1164, 560, 1358},
      {582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708,
       1050, 674, 1244},
      {274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514,
       1050, 708},
      {502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514,
       1278, 480},
      {194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662,
       742, 856},
      {308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320,
       1084, 514},
      {194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274,
       810, 468},
      {536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730,
       388, 1152, 354},
      {502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308,
       650, 274, 844},
      {388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536,
       388, 730},
      {354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342,
       422, 536},
      {468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342,
       0, 764, 194},
      {776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388,
       422, 764, 0, 798},
      {662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536,
       194, 798, 0},
  };
  const int num_vehicles = 4;
  const RoutingIndexManager::NodeIndex depot{0};
};

//! @brief Print the solution.
//! @param[in] data Data of the problem.
//! @param[in] manager Index manager used.
//! @param[in] routing Routing solver used.
//! @param[in] solution Solution found by the solver.
void PrintSolution(const DataModel& data, const RoutingIndexManager& manager,
                   const RoutingModel& routing, const Assignment& solution) {
  int64_t max_route_distance{0};
  for (int vehicle_id = 0; vehicle_id < data.num_vehicles; ++vehicle_id) {
    int64_t index = routing.Start(vehicle_id);
    LOG(INFO) << "Route for Vehicle " << vehicle_id << ":";
    int64_t route_distance{0};
    std::stringstream route;
    while (!routing.IsEnd(index)) {
      route << manager.IndexToNode(index).value() << " -> ";
      const int64_t previous_index = index;
      index = solution.Value(routing.NextVar(index));
      route_distance += routing.GetArcCostForVehicle(previous_index, index,
                                                     int64_t{vehicle_id});
    }
    LOG(INFO) << route.str() << manager.IndexToNode(index).value();
    LOG(INFO) << "Distance of the route: " << route_distance << "m";
    max_route_distance = std::max(route_distance, max_route_distance);
  }
  LOG(INFO) << "Maximum of the route distances: " << max_route_distance << "m";
  LOG(INFO) << "";
  LOG(INFO) << "Problem solved in " << routing.solver()->wall_time() << "ms";
}

void VrpGlobalSpan() {
  // Instantiate the data problem.
  DataModel data;

  // Create Routing Index Manager
  RoutingIndexManager manager(data.distance_matrix.size(), data.num_vehicles,
                              data.depot);

  // Create Routing Model.
  RoutingModel routing(manager);

  // Create and register a transit callback.
  const int transit_callback_index = routing.RegisterTransitCallback(
      [&data, &manager](const int64_t from_index,
                        const int64_t to_index) -> int64_t {
        // Convert from routing variable Index to distance matrix NodeIndex.
        const int from_node = manager.IndexToNode(from_index).value();
        const int to_node = manager.IndexToNode(to_index).value();
        return data.distance_matrix[from_node][to_node];
      });

  // Define cost of each arc.
  routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index);

  // Add Distance constraint. 只有这两个函数跟TSP是不同的，但是没怎么看懂？？
  routing.AddDimension(transit_callback_index, 0, 3000,
                       true,  // start cumul to zero
                       "Distance");
  routing.GetMutableDimension("Distance")->SetGlobalSpanCostCoefficient(100);

  // Setting first solution heuristic.
  RoutingSearchParameters searchParameters = DefaultRoutingSearchParameters();
  searchParameters.set_first_solution_strategy(
      FirstSolutionStrategy::PATH_CHEAPEST_ARC);

  // Solve the problem.
  const Assignment* solution = routing.SolveWithParameters(searchParameters);

  // Print solution on console.
  if (solution != nullptr) {
    PrintSolution(data, manager, routing, *solution);
  } else {
    LOG(INFO) << "No solution found.";
  }
}
}  // namespace operations_research

int main(int /*argc*/, char* /*argv*/[]) {
  operations_research::VrpGlobalSpan();
  return EXIT_SUCCESS;
}

```

### 容量限制
是一种 VRP，车辆承载能力有限，需要在各个地点**取货或交货**。这些物品具有数量（例如重量或体积），而车辆具有的最大容量。问题在于以最低的费用自提或配送商品，同时不超过车辆的容量。

在以下示例中，我们假定提取了所有商品。如果所有项目都在投递，在这种情况下，解决此问题的节目也可以正常运行：在本例中，您可以考虑在车辆完全离开仓库时应用容量限制。但在这两种情况下，容量限制的实现方式相同。

```
// 额外的数据
const std::vector<int64_t> demands{
    0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8,
};
const std::vector<int64_t> vehicle_capacities{15, 15, 15, 15};

// 添加需求回调和容量限制
除了距离回调之外，该求解器还需要一个需求回调（用于返回每个位置的需求）以及一个容量限制维度。以下代码会创建这些内容。
const int demand_callback_index = routing.RegisterUnaryTransitCallback(
    [&data, &manager](const int64_t from_index) -> int64_t {
      // Convert from routing variable Index to demand NodeIndex.
      const int from_node = manager.IndexToNode(from_index).value();
      return data.demands[from_node];
    });
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,    // transit callback index
    int64_t{0},               // null capacity slack
    data.vehicle_capacities,  // vehicle maximum capacities
    true,                     // start cumul to zero
    "Capacity");

与将一对位置作为输入而执行的距离回调不同，需求回调仅依赖于交付的位置 (from_node)。

由于容量限制涉及车辆携带的负载（即车辆沿途累积的重量），因此我们需要为容量创建一个维度，类似于上一个 VRP 示例中的距离维度。

在本例中，我们使用 AddDimensionWithVehicleCapacity 方法，该方法采用容量矢量。

由于此示例中的所有车辆容量都相同，因此您可以使用 AddDimension 方法，该方法为所有车辆数量设置一个上限。但 AddDimensionWithVehicleCapacity 能处理更常见的情况，即不同车辆具有不同的容量。

多种货运类型和容量的问题

在更复杂的 CVRP 中，每辆车可能搭载多种不同类型的货物，每种类型的车辆都有最大容量。例如，燃料运输卡车可能使用多种容量各异的油箱运输多种燃料。为了处理此类问题，只需为每种货运类型创建不同的容量回调和维度（请务必为它们指定唯一名称）。

添加解决方案打印机
解决方案打印机会显示每辆车的路线及其累计负载，即车辆在路线上停止时搭载的总金额。
//! @brief Print the solution.
//! @param[in] data Data of the problem.
//! @param[in] manager Index manager used.
//! @param[in] routing Routing solver used.
//! @param[in] solution Solution found by the solver.
void PrintSolution(const DataModel& data, const RoutingIndexManager& manager,
                   const RoutingModel& routing, const Assignment& solution) {
  int64_t total_distance = 0;
  int64_t total_load = 0;
  for (int vehicle_id = 0; vehicle_id < data.num_vehicles; ++vehicle_id) {
    int64_t index = routing.Start(vehicle_id);
    LOG(INFO) << "Route for Vehicle " << vehicle_id << ":";
    int64_t route_distance = 0;
    int64_t route_load = 0;
    std::stringstream route;
    while (!routing.IsEnd(index)) {
      const int node_index = manager.IndexToNode(index).value();
      route_load += data.demands[node_index];
      route << node_index << " Load(" << route_load << ") -> ";
      const int64_t previous_index = index;
      index = solution.Value(routing.NextVar(index));
      route_distance += routing.GetArcCostForVehicle(previous_index, index,
                                                     int64_t{vehicle_id});
    }
    LOG(INFO) << route.str() << manager.IndexToNode(index).value();
    LOG(INFO) << "Distance of the route: " << route_distance << "m";
    LOG(INFO) << "Load of the route: " << route_load;
    total_distance += route_distance;
    total_load += route_load;
  }
  LOG(INFO) << "Total distance of all routes: " << total_distance << "m";
  LOG(INFO) << "Total load of all routes: " << total_load;
  LOG(INFO) << "";
  LOG(INFO) << "Advanced usage:";
  LOG(INFO) << "Problem solved in " << routing.solver()->wall_time() << "ms";
}
```
- 完整程序
```
#include <cstdint>
#include <sstream>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include "ortools/constraint_solver/routing.h"
#include "ortools/constraint_solver/routing_enums.pb.h"
#include "ortools/constraint_solver/routing_index_manager.h"
#include "ortools/constraint_solver/routing_parameters.h"

namespace operations_research {
struct DataModel {
  const std::vector<std::vector<int64_t>> distance_matrix{
      {0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468,
       776, 662},
      {548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674,
       1016, 868, 1210},
      {776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130,
       788, 1552, 754},
      {696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822,
       1164, 560, 1358},
      {582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708,
       1050, 674, 1244},
      {274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514,
       1050, 708},
      {502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514,
       1278, 480},
      {194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662,
       742, 856},
      {308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320,
       1084, 514},
      {194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274,
       810, 468},
      {536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730,
       388, 1152, 354},
      {502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308,
       650, 274, 844},
      {388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536,
       388, 730},
      {354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342,
       422, 536},
      {468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342,
       0, 764, 194},
      {776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388,
       422, 764, 0, 798},
      {662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536,
       194, 798, 0},
  };
  const std::vector<int64_t> demands{
      0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8,
  };
  const std::vector<int64_t> vehicle_capacities{15, 15, 15, 15};
  const int num_vehicles = 4;
  const RoutingIndexManager::NodeIndex depot{0};
};

//! @brief Print the solution.
//! @param[in] data Data of the problem.
//! @param[in] manager Index manager used.
//! @param[in] routing Routing solver used.
//! @param[in] solution Solution found by the solver.
void PrintSolution(const DataModel& data, const RoutingIndexManager& manager,
                   const RoutingModel& routing, const Assignment& solution) {
  int64_t total_distance = 0;
  int64_t total_load = 0;
  for (int vehicle_id = 0; vehicle_id < data.num_vehicles; ++vehicle_id) {
    int64_t index = routing.Start(vehicle_id);
    LOG(INFO) << "Route for Vehicle " << vehicle_id << ":";
    int64_t route_distance = 0;
    int64_t route_load = 0;
    std::stringstream route;
    while (!routing.IsEnd(index)) {
      const int node_index = manager.IndexToNode(index).value();
      route_load += data.demands[node_index];
      route << node_index << " Load(" << route_load << ") -> ";
      const int64_t previous_index = index;
      index = solution.Value(routing.NextVar(index));
      route_distance += routing.GetArcCostForVehicle(previous_index, index,
                                                     int64_t{vehicle_id});
    }
    LOG(INFO) << route.str() << manager.IndexToNode(index).value();
    LOG(INFO) << "Distance of the route: " << route_distance << "m";
    LOG(INFO) << "Load of the route: " << route_load;
    total_distance += route_distance;
    total_load += route_load;
  }
  LOG(INFO) << "Total distance of all routes: " << total_distance << "m";
  LOG(INFO) << "Total load of all routes: " << total_load;
  LOG(INFO) << "";
  LOG(INFO) << "Advanced usage:";
  LOG(INFO) << "Problem solved in " << routing.solver()->wall_time() << "ms";
}

void VrpCapacity() {
  // Instantiate the data problem.
  DataModel data;

  // Create Routing Index Manager
  RoutingIndexManager manager(data.distance_matrix.size(), data.num_vehicles,
                              data.depot);

  // Create Routing Model.
  RoutingModel routing(manager);

  // Create and register a transit callback.
  const int transit_callback_index = routing.RegisterTransitCallback(
      [&data, &manager](const int64_t from_index,
                        const int64_t to_index) -> int64_t {
        // Convert from routing variable Index to distance matrix NodeIndex.
        const int from_node = manager.IndexToNode(from_index).value();
        const int to_node = manager.IndexToNode(to_index).value();
        return data.distance_matrix[from_node][to_node];
      });

  // Define cost of each arc.
  routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index);

  // Add Capacity constraint.
  const int demand_callback_index = routing.RegisterUnaryTransitCallback(
      [&data, &manager](const int64_t from_index) -> int64_t {
        // Convert from routing variable Index to demand NodeIndex.
        const int from_node = manager.IndexToNode(from_index).value();
        return data.demands[from_node];
      });
  routing.AddDimensionWithVehicleCapacity(
      demand_callback_index,    // transit callback index
      int64_t{0},               // null capacity slack
      data.vehicle_capacities,  // vehicle maximum capacities
      true,                     // start cumul to zero
      "Capacity");

  // Setting first solution heuristic.
  RoutingSearchParameters search_parameters = DefaultRoutingSearchParameters();
  search_parameters.set_first_solution_strategy(
      FirstSolutionStrategy::PATH_CHEAPEST_ARC);
  search_parameters.set_local_search_metaheuristic(
      LocalSearchMetaheuristic::GUIDED_LOCAL_SEARCH);
  search_parameters.mutable_time_limit()->set_seconds(1);

  // Solve the problem.
  const Assignment* solution = routing.SolveWithParameters(search_parameters);

  // Print solution on console.
  PrintSolution(data, manager, routing, *solution);
}
}  // namespace operations_research

int main(int /*argc*/, char* /*argv*/[]) {
  operations_research::VrpCapacity();
  return EXIT_SUCCESS;
}
```


### 车辆自提和配送（Vehicle Routing with Pickups and Deliveries ）
在本部分中，我们将介绍一个 VRP，其中，每辆车在各个位置取走物品，然后将物品放在其他位置。但问题是为车辆分配路线，以方便自提和配送所有商品，同时最大限度地缩短最长路线的长度。

额外提供自提和配送地点。

```
 const std::vector<std::vector<RoutingIndexManager::NodeIndex>>
      pickups_deliveries{
          {RoutingIndexManager::NodeIndex{1},
           RoutingIndexManager::NodeIndex{6}},
          {RoutingIndexManager::NodeIndex{2},
           RoutingIndexManager::NodeIndex{10}},
          {RoutingIndexManager::NodeIndex{4},
           RoutingIndexManager::NodeIndex{3}},
          {RoutingIndexManager::NodeIndex{5},
           RoutingIndexManager::NodeIndex{9}},
          {RoutingIndexManager::NodeIndex{7},
           RoutingIndexManager::NodeIndex{8}},
          {RoutingIndexManager::NodeIndex{15},
           RoutingIndexManager::NodeIndex{11}},
          {RoutingIndexManager::NodeIndex{13},
           RoutingIndexManager::NodeIndex{12}},
          {RoutingIndexManager::NodeIndex{16},
           RoutingIndexManager::NodeIndex{14}},
      };

```

- 完整程序
```
#include <cstdint>
#include <sstream>
#include <vector>

#include "ortools/constraint_solver/routing.h"
#include "ortools/constraint_solver/routing_enums.pb.h"
#include "ortools/constraint_solver/routing_index_manager.h"
#include "ortools/constraint_solver/routing_parameters.h"

namespace operations_research {
struct DataModel {
  const std::vector<std::vector<int64_t>> distance_matrix{
      {0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468,
       776, 662},
      {548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674,
       1016, 868, 1210},
      {776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130,
       788, 1552, 754},
      {696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822,
       1164, 560, 1358},
      {582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708,
       1050, 674, 1244},
      {274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514,
       1050, 708},
      {502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514,
       1278, 480},
      {194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662,
       742, 856},
      {308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320,
       1084, 514},
      {194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274,
       810, 468},
      {536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730,
       388, 1152, 354},
      {502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308,
       650, 274, 844},
      {388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536,
       388, 730},
      {354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342,
       422, 536},
      {468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342,
       0, 764, 194},
      {776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388,
       422, 764, 0, 798},
      {662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536,
       194, 798, 0},
  };
  const std::vector<std::vector<RoutingIndexManager::NodeIndex>>
      pickups_deliveries{
          {RoutingIndexManager::NodeIndex{1},
           RoutingIndexManager::NodeIndex{6}},
          {RoutingIndexManager::NodeIndex{2},
           RoutingIndexManager::NodeIndex{10}},
          {RoutingIndexManager::NodeIndex{4},
           RoutingIndexManager::NodeIndex{3}},
          {RoutingIndexManager::NodeIndex{5},
           RoutingIndexManager::NodeIndex{9}},
          {RoutingIndexManager::NodeIndex{7},
           RoutingIndexManager::NodeIndex{8}},
          {RoutingIndexManager::NodeIndex{15},
           RoutingIndexManager::NodeIndex{11}},
          {RoutingIndexManager::NodeIndex{13},
           RoutingIndexManager::NodeIndex{12}},
          {RoutingIndexManager::NodeIndex{16},
           RoutingIndexManager::NodeIndex{14}},
      };
  const int num_vehicles = 4;
  const RoutingIndexManager::NodeIndex depot{0};
};

//! @brief Print the solution.
//! @param[in] data Data of the problem.
//! @param[in] manager Index manager used.
//! @param[in] routing Routing solver used.
//! @param[in] solution Solution found by the solver.
void PrintSolution(const DataModel& data, const RoutingIndexManager& manager,
                   const RoutingModel& routing, const Assignment& solution) {
  int64_t total_distance{0};
  for (int vehicle_id = 0; vehicle_id < data.num_vehicles; ++vehicle_id) {
    int64_t index = routing.Start(vehicle_id);
    LOG(INFO) << "Route for Vehicle " << vehicle_id << ":";
    int64_t route_distance{0};
    std::stringstream route;
    while (!routing.IsEnd(index)) {
      route << manager.IndexToNode(index).value() << " -> ";
      const int64_t previous_index = index;
      index = solution.Value(routing.NextVar(index));
      route_distance += routing.GetArcCostForVehicle(previous_index, index,
                                                     int64_t{vehicle_id});
    }
    LOG(INFO) << route.str() << manager.IndexToNode(index).value();
    LOG(INFO) << "Distance of the route: " << route_distance << "m";
    total_distance += route_distance;
  }
  LOG(INFO) << "Total distance of all routes: " << total_distance << "m";
  LOG(INFO) << "";
  LOG(INFO) << "Advanced usage:";
  LOG(INFO) << "Problem solved in " << routing.solver()->wall_time() << "ms";
}

void VrpGlobalSpan() {
  // Instantiate the data problem.
  DataModel data;

  // Create Routing Index Manager
  RoutingIndexManager manager(data.distance_matrix.size(), data.num_vehicles,
                              data.depot);

  // Create Routing Model.
  RoutingModel routing(manager);

  // Define cost of each arc.
  const int transit_callback_index = routing.RegisterTransitCallback(
      [&data, &manager](const int64_t from_index,
                        const int64_t to_index) -> int64_t {
        // Convert from routing variable Index to distance matrix NodeIndex.
        const int from_node = manager.IndexToNode(from_index).value();
        const int to_node = manager.IndexToNode(to_index).value();
        return data.distance_matrix[from_node][to_node];
      });
  routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index);

  // Add Distance constraint.
  routing.AddDimension(transit_callback_index,  // transit callback
                       0,                       // no slack
                       3000,  // vehicle maximum travel distance
                       true,  // start cumul to zero
                       "Distance");
  RoutingDimension* distance_dimension =
      routing.GetMutableDimension("Distance");
  distance_dimension->SetGlobalSpanCostCoefficient(100);

  // Define Transportation Requests.
  Solver* const solver = routing.solver();
  for (const auto& request : data.pickups_deliveries) {
    const int64_t pickup_index = manager.NodeToIndex(request[0]);
    const int64_t delivery_index = manager.NodeToIndex(request[1]);
    routing.AddPickupAndDelivery(pickup_index, delivery_index);
    solver->AddConstraint(solver->MakeEquality(
        routing.VehicleVar(pickup_index), routing.VehicleVar(delivery_index)));
    solver->AddConstraint(
        solver->MakeLessOrEqual(distance_dimension->CumulVar(pickup_index),
                                distance_dimension->CumulVar(delivery_index)));
  }

  // Setting first solution heuristic.
  RoutingSearchParameters searchParameters = DefaultRoutingSearchParameters();
  searchParameters.set_first_solution_strategy(
      FirstSolutionStrategy::PARALLEL_CHEAPEST_INSERTION);

  // Solve the problem.
  const Assignment* solution = routing.SolveWithParameters(searchParameters);

  // Print solution on console.
  PrintSolution(data, manager, routing, *solution);
}
}  // namespace operations_research

int main(int /*argc*/, char* /*argv*/[]) {
  operations_research::VrpGlobalSpan();
  return EXIT_SUCCESS;
}
```


### 时间范围限制 VRPTW问题 
许多车辆路线问题都涉及为仅在特定时间段内空闲的客户安排上门服务。

这些问题称为具有时间范围的车辆路线规划问题 (VRPTW)。

VRPTW 示例
本页将通过一个示例来了解如何解决 VRPTW 问题。由于问题涉及时间窗口，因此数据包括时间矩阵，其中包含位置之间的行程时间（而不是先前示例中的距离矩阵）。

下图以蓝色显示要游览的地点，以黑色显示仓库。 每个地点的上方都会显示时间范围。如需详细了解如何定义位置，请参阅 VRP 部分中的位置坐标。

我们的目标是尽可能缩短车辆的总行程时间。

- 完整程序
```
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ortools/constraint_solver/routing.h"
#include "ortools/constraint_solver/routing_enums.pb.h"
#include "ortools/constraint_solver/routing_index_manager.h"
#include "ortools/constraint_solver/routing_parameters.h"

namespace operations_research {
struct DataModel {
  const std::vector<std::vector<int64_t>> time_matrix{
      {0, 6, 9, 8, 7, 3, 6, 2, 3, 2, 6, 6, 4, 4, 5, 9, 7},
      {6, 0, 8, 3, 2, 6, 8, 4, 8, 8, 13, 7, 5, 8, 12, 10, 14},
      {9, 8, 0, 11, 10, 6, 3, 9, 5, 8, 4, 15, 14, 13, 9, 18, 9},
      {8, 3, 11, 0, 1, 7, 10, 6, 10, 10, 14, 6, 7, 9, 14, 6, 16},
      {7, 2, 10, 1, 0, 6, 9, 4, 8, 9, 13, 4, 6, 8, 12, 8, 14},
      {3, 6, 6, 7, 6, 0, 2, 3, 2, 2, 7, 9, 7, 7, 6, 12, 8},
      {6, 8, 3, 10, 9, 2, 0, 6, 2, 5, 4, 12, 10, 10, 6, 15, 5},
      {2, 4, 9, 6, 4, 3, 6, 0, 4, 4, 8, 5, 4, 3, 7, 8, 10},
      {3, 8, 5, 10, 8, 2, 2, 4, 0, 3, 4, 9, 8, 7, 3, 13, 6},
      {2, 8, 8, 10, 9, 2, 5, 4, 3, 0, 4, 6, 5, 4, 3, 9, 5},
      {6, 13, 4, 14, 13, 7, 4, 8, 4, 4, 0, 10, 9, 8, 4, 13, 4},
      {6, 7, 15, 6, 4, 9, 12, 5, 9, 6, 10, 0, 1, 3, 7, 3, 10},
      {4, 5, 14, 7, 6, 7, 10, 4, 8, 5, 9, 1, 0, 2, 6, 4, 8},
      {4, 8, 13, 9, 8, 7, 10, 3, 7, 4, 8, 3, 2, 0, 4, 5, 6},
      {5, 12, 9, 14, 12, 6, 6, 7, 3, 3, 4, 7, 6, 4, 0, 9, 2},
      {9, 10, 18, 6, 8, 12, 15, 8, 13, 9, 13, 3, 4, 5, 9, 0, 9},
      {7, 14, 9, 16, 14, 8, 5, 10, 6, 5, 4, 10, 8, 6, 2, 9, 0},
  };
  const std::vector<std::pair<int64_t, int64_t>> time_windows{
      {0, 5},    // depot
      {7, 12},   // 1
      {10, 15},  // 2
      {16, 18},  // 3
      {10, 13},  // 4
      {0, 5},    // 5
      {5, 10},   // 6
      {0, 4},    // 7
      {5, 10},   // 8
      {0, 3},    // 9
      {10, 16},  // 10
      {10, 15},  // 11
      {0, 5},    // 12
      {5, 10},   // 13
      {7, 8},    // 14
      {10, 15},  // 15
      {11, 15},  // 16
  };
  const int num_vehicles = 4;
  const RoutingIndexManager::NodeIndex depot{0};
};

//! @brief Print the solution.
//! @param[in] data Data of the problem.
//! @param[in] manager Index manager used.
//! @param[in] routing Routing solver used.
//! @param[in] solution Solution found by the solver.
void PrintSolution(const DataModel& data, const RoutingIndexManager& manager,
                   const RoutingModel& routing, const Assignment& solution) {
  const RoutingDimension& time_dimension = routing.GetDimensionOrDie("Time");
  int64_t total_time{0};
  for (int vehicle_id = 0; vehicle_id < data.num_vehicles; ++vehicle_id) {
    int64_t index = routing.Start(vehicle_id);
    LOG(INFO) << "Route for vehicle " << vehicle_id << ":";
    std::ostringstream route;
    while (!routing.IsEnd(index)) {
      auto time_var = time_dimension.CumulVar(index);
      route << manager.IndexToNode(index).value() << " Time("
            << solution.Min(time_var) << ", " << solution.Max(time_var)
            << ") -> ";
      index = solution.Value(routing.NextVar(index));
    }
    auto time_var = time_dimension.CumulVar(index);
    LOG(INFO) << route.str() << manager.IndexToNode(index).value() << " Time("
              << solution.Min(time_var) << ", " << solution.Max(time_var)
              << ")";
    LOG(INFO) << "Time of the route: " << solution.Min(time_var) << "min";
    total_time += solution.Min(time_var);
  }
  LOG(INFO) << "Total time of all routes: " << total_time << "min";
  LOG(INFO) << "";
  LOG(INFO) << "Advanced usage:";
  LOG(INFO) << "Problem solved in " << routing.solver()->wall_time() << "ms";
}

void VrpTimeWindows() {
  // Instantiate the data problem.
  DataModel data;

  // Create Routing Index Manager
  RoutingIndexManager manager(data.time_matrix.size(), data.num_vehicles,
                              data.depot);

  // Create Routing Model.
  RoutingModel routing(manager);

  // Create and register a transit callback.
  const int transit_callback_index = routing.RegisterTransitCallback(
      [&data, &manager](const int64_t from_index,
                        const int64_t to_index) -> int64_t {
        // Convert from routing variable Index to time matrix NodeIndex.
        const int from_node = manager.IndexToNode(from_index).value();
        const int to_node = manager.IndexToNode(to_index).value();
        return data.time_matrix[from_node][to_node];
      });

  // Define cost of each arc.
  routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index);

  // Add Time constraint.
  const std::string time = "Time";
  routing.AddDimension(transit_callback_index,  // transit callback index
                       int64_t{30},             // allow waiting time
                       int64_t{30},             // maximum time per vehicle
                       false,  // Don't force start cumul to zero
                       time);
  const RoutingDimension& time_dimension = routing.GetDimensionOrDie(time);
  // Add time window constraints for each location except depot.
  for (int i = 1; i < data.time_windows.size(); ++i) {
    const int64_t index =
        manager.NodeToIndex(RoutingIndexManager::NodeIndex(i));
    time_dimension.CumulVar(index)->SetRange(data.time_windows[i].first,
                                             data.time_windows[i].second);
  }
  // Add time window constraints for each vehicle start node.
  for (int i = 0; i < data.num_vehicles; ++i) {
    const int64_t index = routing.Start(i);
    time_dimension.CumulVar(index)->SetRange(data.time_windows[0].first,
                                             data.time_windows[0].second);
  }

  // Instantiate route start and end times to produce feasible times.
  for (int i = 0; i < data.num_vehicles; ++i) {
    routing.AddVariableMinimizedByFinalizer(
        time_dimension.CumulVar(routing.Start(i)));
    routing.AddVariableMinimizedByFinalizer(
        time_dimension.CumulVar(routing.End(i)));
  }

  // Setting first solution heuristic.
  RoutingSearchParameters searchParameters = DefaultRoutingSearchParameters();
  searchParameters.set_first_solution_strategy(
      FirstSolutionStrategy::PATH_CHEAPEST_ARC);

  // Solve the problem.
  const Assignment* solution = routing.SolveWithParameters(searchParameters);

  // Print solution on console.
  PrintSolution(data, manager, routing, *solution);
}
}  // namespace operations_research

int main(int /*argc*/, char* /*argv*/[]) {
  operations_research::VrpTimeWindows();
  return EXIT_SUCCESS;
}
```

### 维度

### 资源限制
没看明白

### 惩罚
当没有可行解时，需要添加惩罚。

### 路由选项


