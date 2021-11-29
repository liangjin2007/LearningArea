# 哈工大-苏统华 https://www.bilibili.com/video/BV15E411x7yT?p=1

```
课程内容

Lec 1. Hello CUDA
Lec 2. GPU Hardware Architecture
Lec 3. CUDA Software Abstraction
Lec 4. Memory Hierarchy : 访存
Lec 5. CUDA Debugging & Profiling ： 调试&性能分析
Lec 6. Reduction & Scan ： 算法
```


```
实践为主

4个基本实验
1个创新项目
```


```
参考书

CUDA并行程序设计  GPU编程指南
CUDA专家手册 GPU编程权威指南
```


```
什么是GPU

High throughput computation
High bandwidth memory
```

```
硬件架构

Fermi GF100版子做介绍

16SM * 32 CUDA cores per SM = 512 Core total
8x peak FP64 performance
Direct load/store to memory: hundreds GB/sec

2008 Tesla CUDA: 1.x
2010 Fermi FP64: 2.x
2012 Kepler Dynamic Parallelsim : 3.x能力， GTX 780, GTX680
2014 Maxwell DX12 : 5.x, GTX 980
2016 Pascal Unified Memcpy 3D Memory NVLink
...
```


```
Hello CUDA

vecAdd
```


```
Amdahl's Law

Speedup = xxx

```



```
GPU硬件连接模型

Linking Model: GPU跟CPU的连接模型
Kepler 架构
Fermi 架构
```



