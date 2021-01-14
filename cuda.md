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

- 线程块
```
1. 有共享内存

2. 同步。Kernel中指定的同步点， 在一个块里的线程会被挂起直到它们所有都到达同步点。

3. 线程ID， 是在块之内的线程编号。 根据线程ID可以帮助进行复杂的寻址。
3.1. 二维块 (Dx, Dy)， 线程的索引是(x, y)， 则线程ID=x+y Dx
3.2. 三位块(Dx, Dy, Dz), 线程的索引是(x, y, z)， 则线程ID=x+y Dx+z DxDy

4. 一个块可以包含的线程最大数量是有限的。
```

- 线程块栅栏
```


```
