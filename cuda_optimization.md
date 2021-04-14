## 观点
- [写CUDA到底难在哪？](https://www.zhihu.com/question/437131193/answer/1800559419)

- [Bank Conflict](https://blog.csdn.net/wd1603926823/article/details/78326570)


## CUDA编程 基础与实践 樊哲勇 
- 第五章 获得GPU加速的关键
```
数据传输比例较小
核函数的算术强度较高
核函数中定义的线程数目较多。-> 增大核函数的并行规模
```

- 第六章 CUDA的内存组织
```
内存类型
  静态全局内存 写法 __device__ int c = 1; cudaMemcpyToSymbol。
  常量内存： 有常量缓存的全局内存， 大小64kb。 一个Warp（线程束）中的线程要读取相同的常量内存数据。
  寄存器： 在芯片上。是所有内存中访问速度最高的。有数量限制。一个线程可见。单个线程寄存器数上限255
  局部内存： 寄存器中放不下的变量有可能在局部内存中。 从硬件上看，局部内存只是全局内存的一部分。局部内存的延迟也很高。 每个线程最多512K。
  共享内存：在芯片上。 2070好像是48kb, 2080好像是64kb。对整个线程块可见。主要作用是减少对全局内存的访问，或者改善对全局内存的访问模式。
```
```
SM
  一个GPU包含多个SM。
  一个SM包含如下资源：
    1.一定数量的寄存器： 一般是64k。
    2.一定数量的共享内存： 比如rtx2080是64k，跟寄存器数量一致。
    3.常量内存的缓存
    4.纹理和表面内存的缓存
    5.L1缓存
    6.两个（计算能力6.0）或4个（其他计算能力）线程束调度器warp sheduler。以用在不同线程的上下文间迅速切换。
    7.执行核心
      7.1若干整数运算核心
      7.2若干单精度xxx
      7.3若干双精度xxx
      7.4若干单精度浮点数超越函数
      7.5若干混合精度的张量核心 tensor cores
```
```
一个线程块的最大线程数是1024
一个SM中最多能拥有的线程块个数为16（开普勒架构和图灵架构）或者32（麦克斯韦架构、帕斯卡架构和伏特架构）
一个SM中最多能拥有的线程个数为2048（从开普勒架构到伏特架构）或者1024（图灵架构）， rtx 2070/2080是图灵架构， rtx 3090是安培架构。 一个SM最多有多少个block？
SM中线程的执行是以线程束为单位的，所以最好将线程块大小取为线程束大小（32个线程）的整数倍。
```

```
SM占有率
  要让占有率不小于25%
  并不是占有率越大越好
  
  为啥并行规模足够大，但是SM占有率还是达不到100%
    1.寄存器和共享内存使用量很少的情况。这个情况下占有率取决于线程块大小。
    2.有限的寄存器数量对占有率的约束情况。根据一个SM拥有的线程个数Nt比如等于2048，以及一个SM最多能使用的寄存器数比如64K， 能计算出要把SM线程占满的每个线程最多能使用的寄存器数 64K/2048 = 32。
    3.有限的共享内存对占有率的约束情况。每个SM能激活的线程块数 = 2048/线程块线程数比如128 = 16， 那么每个线程块最多能用的共享内存为48k/16=3k.
```


- 第七章、全局内存的合理使用
```
对全局内存的访问 -> 触发数据传输处理 data transfer -> 一次数据传输处理的数据量在默认情况下是32字节。-> 通过32字节的L2缓存片段cache sector传输到SM。

合并访问
  指的是一个线程束对全局内存的一次访问请求导致最少数量的数据传输。
  一个线程束访问一个全局内存float。 float是4字节， 假设一个线程束有32个线程，那么该线程束将请求32x4 = 128字节的数据。理想情况下将触发128/32=4次L2缓存的数据传输。这是合并访问。
  一次内存读取只能读取地址为0~31字节，32~63字节，64~95字节，96~127字节等片段的数据。
非合并访问
  合并访问的反面
 
数据传输对数据地址的要求是从全局内存转移到L2缓存的一片内存的首地址必须是一个最小粒度（32字节）的整数倍。

cudaMalloc分配的内存的首地址一定是256字节的整数倍，所以一定是32字节的整数倍。

add<<<128,32>>>(x,y,z)

顺序的合并访问：

乱序的合并访问：比如内存索引置换一下，虽然变乱序了，但是从一个线程束的32个线程的整体的角度，但是它访问的内存地址如果合起来依然是0~31字节，32~63字节，或者0~128字节这种的。那么这种是乱序的合并
viod __global__ add_permuted(float* x, float* y, float* z){
  int tid_permuted = threadIdx.x ^ 0x1;
  int n = blockIdx.x*blockDim.x+tid_permuted;
  z[n] = x[n] + y[n];
}

不对齐的非合并访问：
viod __global__ add_permuted(float* x, float* y, float* z){
  int n = blockIdx.x*blockDim.x+threadIdx.x+1;
  z[n] = x[n] + y[n];
}

跨越式的非合并访问：
viod __global__ add_permuted(float* x, float* y, float* z){
  int n = blockIdx.x+blockDim.x*threadIdx.x;
  z[n] = x[n] + y[n];
}

广播式的非合并访问：没看明白
viod __global__ add_permuted(float* x, float* y, float* z){
  int n = threadIdx.x+blockDim.x*blockIdx.x;
  z[n] = x[0] + y[n];
}

```


## 工具的使用
- NSight Compute
https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

- 更细的一些概念
```
SP: Streaming Processor， 即核心，可以跑一个线程。
Warp: 一般32个线程，硬件上来说的话即32个SP。
SM:GPU大核
  多个SP
  多个sheduler
  寄存器
  share memory
  等

https://baijiahao.baidu.com/s?id=1614489586434858736&wfr=spider&for=pc
RTX 2070 SMs数: 36
cuda cores: 2304
可以算出一个SM是64个core, 也就是2个warp。因为一个SM的最大可用寄存器数为64k。
```

- cycle
```

```


- NSight System
```
1. 项目属性右键设置debug
2. VS2017 -> 扩展 -> NSight - > Start CUDA Debugging(Next-Gen)
VS2017 -> 扩展 -> NSight -> Windows -> Warp Info
VS2017 -> 扩展 -> NSight -> Windows -> Lane Info ? 

```
