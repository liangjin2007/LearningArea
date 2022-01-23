# ComputerAnimationAlgorithmsAndTechniques.3rdEdition
## Chapter 1 Introduction
## Chapter 2 Technical Background
- 2.1.空间和变换
- 2.2.定向表示
```
Fixed-angle representation
Euler angle representation
  Local coordinate system Yaw, Pitch, Roll
Angle and axis representation
Quaternion representation
Exponential map representation
  (x*cos(angle), y*cos(angle), z*cos(angle))
```
## Chapter 3 Interpolating Values
- 3.1.插值Interpolation
- 3.2.沿着曲线控制点的运动
```
曲线参数化
3.2.1.计算弧长
  弧长参数化 
  估计弧长
    数学上应该要从微分du = ds/sqrt(x'(u)^2 + y'(u)^2 + z'(u)^2)，求出u=u(s) 然后 P(u)可以表示为Q(s) = P(u(s))
    数值方法：
      参数 vs 弧长 表： 离散化参数空间u, 求出每个indexed position对应的弧长by prefixsum(segment_i)， 前向差分，就得到一张弧长表。
      这张表怎么用：
    数值积分
      Simpson积分
      梯形积分
3.2.2.速度控制 Speed control
  弧长关于t的函数 s = s(t)， 弧长后面又被称为distance。
  
```


## Appendix A
```
Double Buffering
Compositing
  over operation
  interleave交错 动词
  alpha channel
  erroneous错误的
  A-buffer algorithm提供
  rgbaz
Motion blur
  temporal aliasing
Drop shadows  
Billboarding and impostors
```
## Appendix B
- Vectors and matrices
```
运算
  乘法
  转置
  identity matrix
  inverse
  determinant
  submatrix
Inerse matrix and solving linear systems
  inverse transformation
  LU decomposition
  lower triangular matrix x upper triangular matrix
Singular value decomposition
```
- Geometric computation
```  
position, point
direction
vector
transformation
Components of a vector i,j,k     A = a i + b j + c k
Length of a vector
Dot product of two vectors
Cross product of two vectors
Closest point between two lines in three-space
Area of a triangle
Area of a polygon
  signed areas of simple elements
The cosine rule
  C^2 = A^2 + B^2 - 2 A B cos theta
Barycentric coordinates
Computing bounding shapes
  bounding box
  bounding slabs
  bounding spheres
  convex hull
```

- Transformations
```  
Transforming a point using vector-matrix multiplications
Transforming a vector using vector-matrix multiplications
axis-angle rotations
quaternions
  rotations by quaternions
  
```

- Denevit and Hartenberg representation for linked appendages
```

```
