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

- 计算兼容性

- 多设备： 类型必须一样。

- 模式切换
```
primary surface
```
## 第四章 API

- 一个C语言的扩展
```
C语言扩展集
runtime
```

- 语言扩展
```
1. 函数类型限定句
__device__在设备上执行，仅可从设备调用
__global__声明一个函数作为一个存在的kernel。 在设备执行的，仅可从主机调用。
__host__ 在主机上执行，仅可从主机调用
__global__ __device__
  不支持递归
  不能声明静态变量
  ？？ 不能有自变量的一个变量数字
__device__
  不能取得函数地址
  函数指向__global__函数是支持的
不能一起使用__global__和__host__
__global__函数必须有void的返回类型。
任何调用到一个__global__函数必须指定它的执行配置。
？？ 对一个__global__函数的调用是同步的
__global__函数参数目前是通过共享内存到设备的，并且被限制在256字节。

2. 变量类型限定句
__device__声明驻留在设备上的一个变量， 全局内存空间，具有应用的生存期， 从栅格内所有线程和从主机通过runtime库是可访问的。
  以下与__device__一起使用：
    __constant__: 驻留在常量内存空间，具有应用的生存期，从栅格内所有线程和从主机通过runtime库是可访问的。
    __shared__ : 驻留在线程块的共享内存空间中，具有块的生存期，只有块之内的所有线程是可访问的。

3. 执行配置： 新的指令指定kernel如何在设备上执行
<<<Dg, Db, Ns, S>>>
Dg栅格维度 Dg.x, Dg.y
Db 块维度Db.x, Db.y, Db.z
Ns: 静态分配的内存之外的动态分配每个块的内存， 默认0，可选
S: Stream相关， 默认0， 可选


4. 内置变量指定栅格和块的维数， 还有块和线程的ID
gridDim: dim3 栅格维度
blockIdx: uint3 块索引
blockDim: dim3  块维度
threadIdx: uint3包含块之内的线程索引


5. nvcc编译 
__noinline__
行程计数 #pragma unroll 5
```

- 公共Runtime组件 可同时被Host和Device调用
```
1.内置矢量类型 float4, ...
2.dim3 = uint3
3.数学函数，看附录B
4.时间函数clock_t clock();
5.纹理类型， texture reference, texture fetch
Texrure<Type, Dim, ReadMode> texRef;
  cudaReadModeNormalizedFloat : unsigned 的整型类型被映射到[0.0，1.0]，signed 的整型类型被映射到[-1.0，1.0]
  cudaReadModeElementType
纹理坐标是否是Normalized : Normalized 的纹理通过坐标[0.0，1.0)引用，而不是[0，N)
```

- 设备Runtime组件 只能用于设备函数
```
1. 更快的数学函数版本，比如__sin(x)
2. 同步函数 __syncthreads(); 在一个块内同步所有线程。一旦所有线程到达了这点，恢复正常执行。
3. 原子函数 atomicAdd()
4. 数组纹理操作tex1D, tex2D; 设备内存纹理操作
5. Type Casting: __int_as_float(), __float_as_int()
6. 类型转换函数： __float2int_[rn,rz,ru,rd](), __int2float_[](), 

```

- 主机Runtime组件 只能被主机函数使用
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
这些API是互斥的，一个应用程序应该选择其中之一来使用


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

3. OpenGL互操作
OpenGL 缓冲器对象可以被映射到CUDA 地址空间，使CUDA 能够读取被OpenGL 写入的数据，或者使CUDA 能够写入被OpenGL 消耗的数据。

4. Direct3D互操作
Direct3D 9.0 顶点缓冲器可以被映射到CUDA 地址空间，使CUDA 能够读取被Direct3D 写入的数据，或者使CUDA 能够写入被Direct3D 消耗的数据
一个CUDA context 每次只可以互用一个Direct3D 设备，通过把begin/end 函数括起来调用。
CUDA context 和Direct3D 设备必须建立在同一个GPU 上。可以通过查询与CUDA 设备是否关联使用Direct3D的适配器来确保。对于runtime API 使用cudaD3D9GetDevice()（参见附录D.9.7），对于驱
动API 使用cuD3D9GetDevice()。

5. 异步并发执行


```



## 第五章 性能指导

## 第六章 矩阵乘法的例子

## 附录A 技术规格
```
 一个块最大线程数是512；
 一个线程块在x-，y-，和z-空间的最大大小分别是512，512 ，和64；
 一个线程块栅格的每个最大空间大小是65535；
 Warp 的大小是32 个线程
 每个多处理器的寄存器数量是8192；
 每个多处理允许的共享内存大小是16KB，被分为16 个bank；
 常驻内存大小是64KB；
 每个多处理器常驻内存可用的缓存是8KB；
 每个多处理器纹理内存可用的缓存是8KB；
 每个多处理器最大可用的块数量是8；
 每个多处理器最大可用的warp 数量是24；
 每个多处理器最大可用的线程数是768；
 对于绑定到一维CUDA 数组的texture reference，最大宽度是2
13
15
 对于绑定到二维CUDA 数组的texture reference，最大宽度是2
16
，最大高度是2
 对于绑定到线性内存的texture reference ，最大宽度是2
27
 Kernel 大小限制为2 百万个原生指令集；
 每个多处理器由8 个处理器构成，因此每个多处理器可以在四个时钟周期内处理一个32 个线程的
warp。
```


## 附录B 数学函数

## 附录C 原子函数

## 附录D Runtime API 参考 page 91-136

# QA

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

# CUDA编程之快速入门
https://www.cnblogs.com/skyfsm/p/9673960.html


