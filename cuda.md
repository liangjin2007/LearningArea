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

- 硬件实现
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

```
