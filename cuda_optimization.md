## 观点
- [写CUDA到底难在哪？](https://www.zhihu.com/question/437131193/answer/1800559419)

- [Bank Conflict](https://blog.csdn.net/wd1603926823/article/details/78326570)
```
比如有个global memory buffer 是三角形顶点坐标{{p0_i, p1_i, p2_i}, ...} 它的尺寸为3*4*3*numTriangles。如何优化内存？

```
- CUDA编程 基础与实践 樊哲勇 
```
第五章 获得GPU加速的关键
数据传输比例较小
核函数的算术强度较高
核函数中定义的线程数目较多。-> 增大核函数的并行规模
```
```
第六章 CUDA的内存组织
静态全局内存 写法 __device__ int c = 1; cudaMemcpyToSymbol。
常量内存： 有常量缓存的全局内存， 大小64kb。 一个Warp（线程束）中的线程要读取相同的常量内存数据。
寄存器： 是所有内存中访问速度最高的。有数量限制。一个线程可见
局部内存： 寄存器中放不下的变量有可能在局部内存中。 从硬件上看，局部内存只是全局内存的一部分。局部内存的延迟也很高。 每个线程最多512K。
共享内存：
```
## 工具的使用
- NSight Compute

- NSight System
```
1. 项目属性右键设置debug
2. VS2017 -> 扩展 -> NSight - > Start CUDA Debugging(Next-Gen)
VS2017 -> 扩展 -> NSight -> Windows -> Warp Info
VS2017 -> 扩展 -> NSight -> Windows -> Lane Info ? 

```
