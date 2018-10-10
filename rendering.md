# 图形管线 Graphics Pipeline

- 架构 Architecture
   - 管线？例子？理解并行执行？单个人干多个事情？no 多个人干多个事情
   - 渲染衡量标准? rendering speed?FPS
   - CPU上的应用层，多线程，应用层算法，比如遮挡剔除
   - 几何处理，GPU，变换，投影
   - 光栅化，从几何找像素
   - 像素处理，决定像素颜色

- 应用层 Application Stage
   - 碰撞检测 collision detection
- 几何处理 Geometry Processing
   - 顶点着色vertex shading
      - 顶点着色，其实这一阶段已经超出它的名字所能干的事情，甚至都不用碰着色方程
      - coordinate system
         - model space, model coordinate
         - model transform
         - world space, world coordinate
         - view transform
         - camera space, view space or eye space, 
      - projection
         - 视景体view volume
            - 正交投影为矩形
            - 透视投影为frustum 截断金字塔
         - 投影的结果为一个unit cube 称为 canonical view volume， 称为裁剪坐标clip coordinate（齐次坐标）
         - 正交投影orthographic
         - 透视投影perspective
      
      - clipping
         - 输入是clip coordinate
   - 可选顶点处理Optional vertex processing
      - tessellation
         - hull shader
         - tessellator
         - domain shader
      - geometry shading
         - 粒子生成particle generation
         - generate a square(two triangles) from a point
      - stream output
         - generate vertex arrays for CPU usage or GPU usage
   - screen mapping
   - 
- 光栅化 Rasterization
- 像素处理 Pixel Processing
- 整个管线

# Texturing

- Texturing Pipeline


# Light Map

- Reference 
   - [https://github.com/TheRealMJP/BakingLab](https://github.com/TheRealMJP/BakingLab)
   - [https://github.com/ands/lightmapper](https://github.com/ands/lightmapper)
   
- 🦐BB

# Irradiance Volume

- Reference
[https://github.com/pyalot/webgl-deferred-irradiance-volumes](https://github.com/pyalot/webgl-deferred-irradiance-volumes)
