# cuda docs homepage https://docs.nvidia.com/cuda/
## cuda programming guides
### cuda programming guide https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
### cuda toolkit documentation https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#syntax
### CUDA C Best Practice
### cuda occupacy calculator https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html


# CUDA工具
- NSight Compute
- NSight System

# 《CUDA_Programming Guide 1.1中文版》笔记

## 第一章 CUDA介绍
Control, Cache, ALU, DRAM, Shared memory.
## 第二章 编程模型
- 线程批处理
```
Host, Device
Host write Kernel
Device assign Grid/Block/Thread
```

- 线程块Thread Block
```
1. 有共享内存

2. 同步。Kernel中指定的同步点， 在一个块里的线程会被挂起直到它们所有都到达同步点。

3. 线程ID， 是在块之内的线程编号。 根据线程ID可以帮助进行复杂的寻址。
3.1. 二维块 (Dx, Dy)， 线程的索引是(x, y)， 则线程ID=x+y Dx
3.2. 三位块(Dx, Dy, Dz), 线程的索引是(x, y, z)， 则线程ID=x+y Dx+z DxDy

4. 一个块可以包含的线程最大数量是有限的。
```

- 线程块栅栏Grid
```
1.线程协作的减少会造成性能损失，因为来自同一个栅格的不同线程块中的线程彼此之间不同通讯和同步。

2.块ID： 只有二维？(Dx, Dy), 块的索引是(x, y)， 则块ID=x+y Dx

```

- 内存模型: 数据读写
```
1. 读写每个线程的寄存器Register
2. 读写每个线程的本地内存 local memory
3. 读写每个块的共享内存 shared memory
4. 读写每个栅格的全局内存 global memory
5. 读每个栅格的常量内存 constant memory
6. 读每个栅格的纹理内存 texture memory
```

## 第三章 硬件实现

？？ 难点在于对warp的理解

- SIMD多处理器
```
1.设备
2.多处理器：包含很多个处理器
3.设备=多处理器的数组
4.每个多处理器使用SIMD架构，每个处理器执行同一指令，但操作不同的数据。
5.每个多处理器的内存模型：
5.1.每个处理器有一组本地32位寄存器
5.2.并行数据缓存或共享内存，被所有处理器共享，实现内存共享。
5.3.
```

- 执行模式
```
？？ 线程块在一个批处理中被一个多处理器执行，被称为active。 每个active块被划分称为SIMD线程组，称为warps（对应于栅格，每个warp对应于线程块，包括数量相同的线程，叫做warp大小）
每个线程可使用的寄存器数量 = 每个多处理器寄存器总数除以并发的线程数量。
这里的块跟线程块又有区别？
一个块内的warp次序是未定义的。

在一个线程块栅格内的块次序是未定义的，并且在块之间不存在同步机制，因此来自同一个栅格的二个不同块的线程不能通过全局内存彼此安全地通讯。
```

- 五、主机Runtime组件 只能被主机函数使用
```
它提供函数来处理：
  设备管理
  Context管理
  内存管理
  编码模块管理
  执行控制
  Texture reference管理
  OpenGL和Direct3D的互用性
  
它由二个API组成：
一个低级的API调用CUDA驱动程序API    函数以cu开头   通过cuda动态库提供
一个高级的API调用CUDA runtime API  函数以cuda开头 通过cudart动态库提供
```

这些API是互斥的，一个应用程序应该选择其中之一来使用
- 五、一、公共概念
```
概念
1.设备
一个主机线程只能在一个设备上执行设备代码。因此，多主机线程需要在多个设备上执行设备代码。另外，任何在一个主机线程中通过runtime创建的CUDA 源文件不能被其它主机线程使用。


2. 内存
设备内存可被分配到线性内存或者是CUDA 数组。
在设备上的线性内存使用32-bit 地址空间，因此单独分配的实体可以通过指针的互相引用，例如，在一个二元的树结构中。

CUDA 数组是针对纹理拾取优化的不透明的内存布局。它们是一维或二维的元素组成的，每个有1 个，2个或者4 个组件，每个组件可以是有符号或无符号8-，16- 或32-bit 整型，16-位浮点(仅通过CUDA 驱动
程序API 支持)，或32 位浮点。CUDA 数组只能通过kernel 纹理拾取读取。

通过主机的内存复制函数，线性内存和 CUDA 数组都是可读和可写的。

不同于由malloc()函数分配的pageable 主机内存，主机runtime 同样提供可以分配和释放page-locked主机内存的函数。
如果主机内存被分配为page-locked ，使用page-locked 内存的优势是，主机内存和设备内存之间的带宽将非常高。但是，分配过多的page-locked 内存将减少系统可用物理内存的大小，从而降低系统整体的性能。
```

- 五、二、Runtime API
```
1.Runtime初始化
2.设备管理
cudaGetDeviceCount
cudaGetDeviceProperties
cudaDeviceProp结构体
cudaSetDevice
3.内存管理
用来分配和释放设备内存，访问在全局内存中任意声明的变量分配的内存，和从主机内存到设备内存之间的数据传输。
cudaMalloc, cudaMallocPitch, cudaFree, cudaMemcpy, cudaMemcpy3DParams, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice
4.流管理
cudaStream_t
cudaStreamCreateWithFlags
cudaStreamSynchronize
5.事件管理
6.纹理管理
7.opengl互操作
8.direct3d互操作
9.使用设备仿真方式调试
```

- 五、三、驱动API
```
驱动API是基于句柄的，命令式的API，多数对象通过不透明的句柄引用。
1.初始化 
cuInit
2.设备管理
cuDeviceGetCount, cuDeviceGet
3.Context管理
cuCtxCreate, cuCtxAttach, cuCtxDetach
4.模块管理
cuModuleLoad, cuModuleGetFunction
5.执行控制
cuFuncSetBlockShape
cuLaunchGrid
6.内存管理
cuMemAlloc
7.流管理
8.事件管理
9.纹理reference管理
10.opengl互操作性
11.Direct3D互操作性
```


## Bank Conflict
```
简介

　　目前 CUDA 装置中,每个 multiprocessor 有 16KB 的 shared memory。Shared memory 分成 16 个 bank。如果同时每个 thread 是存取不同的 bank,就不会产生任何问题,存取shared memory 的速度和存取寄存器相同。不过,如果同时有两个(或更多个) threads 存取同一个 bank 的数据,就会发生 bank conflict,这些 threads 就必须照顺序去存取,而无法同时存取 shared memory 了。

例子

　　　Shared memory 是以 4 bytes 为单位分成 banks。因此,假设以下的数据:　__shared__ int data[128];

那么,data[0] 是 bank 0、data[1] 是 bank 1、data[2] 是 bank 2、...、data[15] 是 bank15,而 data[16] 又回到 bank 0。由于 warp 在执

行时是以 half-warp 的方式执行,因此分属于不同的 half warp 的 threads,不会造成 bank conflict。

因此,如果程序在存取 shared memory 的时候,使用以下的方式:　int number = data[base + tid];

那就不会有任何 bank conflict,可以达到最高的效率。

　　　但是,如果是以下的方式:int number = data[base + 4 * tid];那么,thread 0 和 thread 4 就会存取到同一个 bank,thread 1 和 

thread 5 也是同样,这样就会造成 bank conflict。在这个例子中,一个 half warp 的 16 个 threads 会有四个 threads 存取同一个 bank,

因此存取 share memory 的速度会变成原来的 1/4。一个重要的例外是,当多个 thread 存取到同一个 shared memory 的地址

时,shared memory可以将这个地址的 32 bits 数据「广播」到所有读取的 threads,因此不会造成 bank conflict。例如:int number = 

data[3];　这样不会造成 bank conflict,因为所有的 thread 都读取同一个地址的数据。很多时候 shared memory 的 bank conflict 可以

透过修改数据存放的方式来解决。例如,

以下的程序:data[tid] = global_data[tid];

　　　　　　　　　...

　　　　　 int number = data[16 * tid];

会造成严重的 bank conflict,为了避免这个问题,可以把数据的排列方式稍加修改,把存取方式改成:

int row = tid / 16;

int column = tid % 16;

data[row * 17 + column] = global_data[tid];

...

int number = data[17 * tid];

这样就不会造成 bank conflict 了。


疑问

　　　为什么 shared memory 存在 bank  conflict，而 global memory 不存在？因为访问 global memory 的只能是 block，而访问 shared memory 的却是同一个 half-warp 中的任意线程。
```

```
warp id and lane id : https://stackoverflow.com/questions/44337309/whats-the-most-efficient-way-to-calculate-the-warp-id-lane-id-in-a-1-d-grid

CUDA中grid、block、thread、warp与SM、SP的关系
首先概括一下这几个概念。其中SM（Streaming Multiprocessor）和SP（streaming Processor）是硬件层次的，其中一个SM可以包含多个SP。thread是一个线程，多个thread组成一个线程块block，多个block又组成一个线程网格grid。

现在就说一下一个kenerl函数是怎么执行的。一个kernel程式会有一个grid，grid底下又有数个block，每个block是一个thread群组。在同一个block中thread可以通过共享内存（shared memory）来通信，同步。而不同block之间的thread是无法通信的。

CUDA的设备在实际执行过程中，会以block为单位。把一个个block分配给SM进行运算；而block中的thread又会以warp（线程束）为单位，对thread进行分组计算。目前CUDA的warp大小都是32，也就是说32个thread会被组成一个warp来一起执行。同一个warp中的thread执行的指令是相同的，只是处理的数据不同。

基本上warp 分组的动作是由SM 自动进行的，会以连续的方式来做分组。比如说如果有一个block 里有128 个thread 的话，就会被分成四组warp，第0-31 个thread 会是warp 1、32-63 是warp 2、64-95是warp 3、96-127 是warp 4。而如果block 里面的thread 数量不是32 的倍数，那他会把剩下的thread独立成一个warp；比如说thread 数目是66 的话，就会有三个warp：0-31、32-63、64-65 。由于最后一个warp 里只剩下两个thread，所以其实在计算时，就相当于浪费了30 个thread 的计算能力；这点是在设定block 中thread 数量一定要注意的事！

一个SM 一次只会执行一个block 里的一个warp，但是SM 不见得会一次就把这个warp 的所有指令都执行完；当遇到正在执行的warp 需要等待的时候（例如存取global memory 就会要等好一段时间），就切换到别的warp来继续做运算，借此避免为了等待而浪费时间。所以理论上效率最好的状况，就是在SM 中有够多的warp 可以切换，让在执行的时候，不会有「所有warp 都要等待」的情形发生；因为当所有的warp 都要等待时，就会变成SM 无事可做的状况了。

实际上，warp 也是CUDA 中，每一个SM 执行的最小单位；如果GPU 有16 组SM 的话，也就代表他真正在执行的thread 数目会是32*16 个。 不过由于CUDA 是要透过warp 的切换来隐藏thread 的延迟、等待 ，来达到大量平行化的目的 ，所以会用所谓的 active thread 这个名词来代表一个SM 里同时可以处理的thread 数目。而在block 的方面， 一个SM 可以同时处理多个thread block，当其中有block 的所有thread 都处理完后，他就会再去找其他还没处理的block 来处理 。假设有16 个SM、64 个block、每个SM 可以同时处理三个block 的话，那一开始执行时，device 就会同时处理48 个block； 而剩下的16 个block 则会等SM 有处理完block 后，再进到SM 中处理，直到所有block 都处理结束
```


