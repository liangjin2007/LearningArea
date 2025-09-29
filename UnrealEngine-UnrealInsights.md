## 文档
- https://dev.epicgames.com/documentation/zh-cn/unreal-engine/unreal-insights-in-unreal-engine#trace


## 分析内存
Build.cs中需要添加？
"TraceLog",
"TraceAnalysis",
"TraceServices",
"TraceInsights",
"TraceInsightsCore"


- 命令行参数-trace
VisualStudio 项目的调试中添加 -trace=memory,insights， 编辑器起来的时候即启动trace.
