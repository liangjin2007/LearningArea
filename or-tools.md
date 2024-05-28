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

打包问题有多种类型。其中最重要的两个是“背包问题”和“装箱问题”。

- 背包问题：

在简单的背包问题中，只有一个容器（背包）。这些项具有价值和尺寸，目标是打包具有最高总价值的部分项。

对于价值等于大小的特殊情况，目标是使打包商品的总大小最大化。

多维背包问题，此类问题是指物品有多个物理数量（例如重量和体积），背包对每个数量都有容量。这里，术语不一定是指高度、长度和宽度的常规空间维度。但是，一些问题可能涉及空间维度，例如，寻找将矩形箱打包到矩形存储箱中的最佳方法。

多个背包问题，此类问题涉及多个背包，目标是使所有背包中已打包的物品的总价值最大化。

请注意，您可能是针对单个背包的多维问题，也可能是只有一个维度的多维背包问题。































