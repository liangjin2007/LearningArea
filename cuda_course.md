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

第一种
CPU <-> Northbridge memory controller <-> CPU memory
Northbridge<->PCI Express<-> GPU Memory Controller<-> GPU

...
```

```
CPU: Latency Oriented Cores
GPU: Throughput Oriented Cores

```


```
CUDA抽象机制

CUDA作为并行计算平台
语言： CUDA C
编辑器 ： Visual Studio
编译 ： nvcc
SDK : CUDA toolkit, Libraries, Samples
Profiler & Debugger : NSight


CUDA作为编程模型
GPU硬件的软件抽象
不依赖于操作系统，CPU, NVIDIA GPUs


thread  <-- CUDA cores(register, program counter,  state)
block   <- streaming multiprocessors(threads, shared memory)
非抢占的方式

并发线程模型
Thread->Block->Grid

内存模型
registers<--thread-->local memory
shared memory <--block --> local memory
global memory<-- grid
每个线程：
  能读写每线程的寄存器
  能读写每线程的局部内存
  能读写每块的共享内存
  能读写每gird的全局内存
  能读每grid的constant memory


同步机制
  原子函数： 比较慢。
```


```
CUDA函数
```

```
内核线程索引
```

```
设备占用率 #active warps / #max warps

计算能力
threads per block
registers per thread
shared memory per block and shared memory configuration
```


```
线程束的调度机制

Block调度器
Ready Queue : 先都在Ready Queue里
Executing 执行 ： 
Suspended 挂起 ：如果有某warp有访存，该warp就会被挂起。 
Memory Request Pending : 

验证warp的线程数量，看看是不是32
加入计时功能，对warp的调度时间进行输出，并绘出散点图进行分析。
变大block和grid的大小会如何

内置变量warp size
```


```
CUDA
```
