# Real-Time Rendering 4th Edition
- 1. 图形渲染管线
- 2. GPU
- 3. 变换
- 4. 着色基础
- 5. 贴图
- 6. 阴影
- 7. 光照和颜色
- 8. 基于物理着色
- 9. 局部光照
- 10. 全局光照
- 11. 图像空间效果
- 12. 多边形之外
- 13. 体素和半透明渲染
- 14. 非真实感渲染
- 15. 多边形技术
- 16. 曲线和曲面
- 17. 管线优化
- 18. 加速算法
- 19. 高效着色
- 20. 虚拟现实和增强现实
- 21. 相交测试方法
- 22. 图形硬件
- 23. 将来

# 图形管线 Graphics Rendering Pipeline

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
   
   - 裁剪clipping
      - 输入是clip coordinate齐次坐标，用于正确地处理插值和裁剪
      - 三角形部分在视景体外时，需要裁剪
      - 透视除法perspective division
      - 输出归一化设备坐标 normalized device coordinates
   
   - 屏幕映射screen mapping
      - 将归一化设备坐标变换到 window coordinate (x,y,z), z in [0,1] or [-1,1]
      - 屏幕坐标screen coordinate(x,y)
      - pixel index = floor(c)
      - pixel center coordinate = pixel index + 0.5
      
   - model coordinate -> world coordinate -> view coordinate -> clip coordinate(4d) -> normalized device coordinate(3d) -> window coordinate
- 光栅化 Rasterization
   - 找pixels in 三角形
   - 又称为scan conversion
   - 判断pixel center是否在三角形内
   - 或者用更多的采样， supersampling, multisampling aliasing
   - 具体步骤
      - Triangle Setup
         - differentials
         - edge equations
      - Triangle Traversal
         - 找到哪些pixel在三角形内，并生成fragment
         - 透视正确的插值出fragment的属性
   
- 像素处理 Pixel Processing
   - 输入是fragments
   - 分两步：
      - 像素着色pixel shading
         - 为每个像素生成一个或者多个color, 即fragment color
         - pixel shader or fragment shader
            - texturing
               - 把一个或者多个图像粘到对象上
      - 融合merging
         - 谁跟谁融合？
            - fragment color
            - color in color buffer
         - 称为 ROP raster operations(pipeline)或者render output unit或者blend operations
         - 不可编程，但是可以配置，也可以实现各种各样的效果
         - 解决visibility
            - z buffer or depth buffer
            - Transparency是缺点
         - alpha channel
         - stencil buffer
         - framebuffer 包含所有的buffer
         - double buffering双缓存
            - back buffer
            - front buffer
         
- 整个管线

# The Graphics Processing Unit

- ![](https://github.com/liangjin2007/data_liangjin/blob/master/graphicspipeline.jpg?raw=true,"graphicspipeline")

- ![](https://github.com/liangjin2007/data_liangjin/blob/master/graphicshardwaretimeline.jpg?raw=true,"timeline")

- Vertex Shader
   - ![](https://github.com/liangjin2007/data_liangjin/blob/master/vertexshader.jpg?raw=true,"vertexshader")
   - instancing
   - vertex blending
   - silhouette rendering
   - skinning
   - morphing
   - procedural deformation, flags, cloth, water
   - particle creation
   - lens distortion
   - heat haze
   - water ripples
   - page curls
   - terrain height field
   
- Tessellation Stage
   - ![](https://github.com/liangjin2007/data_liangjin/blob/master/tessellation.jpg?raw=true,"tessellation")
   - hull shader, tessellator, etc
- Geometry Shader
   - ![](https://github.com/liangjin2007/data_liangjin/blob/master/geometryshader.jpg?raw=true,"geometryshader")
- Pixel Shader
   - ![](https://github.com/liangjin2007/data_liangjin/blob/master/pixelshader.jpg?raw=true,"pixelshader")
   - multiple render targets MRT, 4个或8个
   - render target
   - deferred shading
   - UAV
   - SSBO
   - ROVs
- Merging Stage
   - early-z
- Compute Shader
   - ![](https://github.com/liangjin2007/data_liangjin/blob/master/computershader.jpg?raw=true,"computeshader")
   - GPU to GPU
   - distribution, average luminance twice faster than pixel shader
   - particle system
   - facial animation
   - culling
   - image filtering
   - improving depth precision
   - shadows
   - depth of field

# Transforms
- Matrix, Basic Transforms, Special Matrix, Quaternions, Vertex Blending
- Morphing
```
linear blending
morph target: N + sum(wi blendshape_i)
pose-space deformation: combine vertex blending and morph target
precomputed vertex textures
hardware supporting stream-out
tessellation stage

```
- Geometry Cache Playback
   - quantization
   - spatial prediction
   - temporal prediction
   - compression
- Projections
# 着色基础 Shading Basics
- 着色模型Shading Models
```
决定曲面颜色based on view direction, light direction and surface orientation(normal)
n, v, l
```
   - Gooch shading model : 
   ```
   c_shaded = s c_hightlight + (1-s)(t c_warm + (1-t) c_cool)
   c_cool = (0, 0, 0.55) + 0.25 c_surface
   c_warm = (0.3, 0.3, 0) + 0.25 c_surface
   c_highlight = (1, 1, 1)
   t = ((n.l)+1)/2
   r = 2(n.l)n - l
   s = clamp(100(r.v)-97)
   ```
- 光源Light Sources
- 实现细节
- 走样和反走样 aliasing and antialiasing
   - MSAA
- 透明，alpha，复合 transparency, alpha, compositing
   - depth peeling
- 显示编码display encoding
   - gamma correction

# Texturing
利用图像，函数，或者其他资源，给曲面每个位置分配一个颜色
例子，砖brick，砂浆mortar
两个三角形
砖是光泽的glossy
砂浆是不光滑的matte
如何修改粗糙度roughness，使用第二张图像
砖太平了flat, 使用bump mapping
parallex mapping
parallex occlusion
displacement mapping
多图像texturing
![coloring and bump mapping](https://github.com/liangjin2007/data_liangjin/blob/master/texturing.jpg?raw=true)

- Texturing Pipeline
修改着色方程shading equation中的值。
texels
修改颜色值
修改normal
修改roughness

- ![texturing pipeline](https://github.com/liangjin2007/data_liangjin/blob/master/texturepipeline.jpg?raw=true)

- projection function
   - cylinder, planar, spherical, polymap, parameterization, unwrap

- corresponder function
   - wrap, repeat or tile
   - mirror
   - clamp or clamp to edge
   - border or clamp to border
   
- get texture value
   - memory lookup
   - computation of a function
   
- Image Texturing
   - magnification
   - minification
      - antialiasing: mipmap
      - overblurring: summed-area table
      - Unconstrained Anisotropic Filtering
   - volume texture
   - cube map
   - texture representation
      - texture compression
      - texture atlas, texture arrays, bindless textures
   - texture compression
      - ETC
      - BC1-BC5
      - BC6H
      - BC7
      - ETC2
      - EAC
      - DXT5
      - PVRTC
      - ASTC
- Procedural Texturing
   - volume textures
   - noise function
      - octave
      - turbulence
- Texture Animation
   - changing texture coordinates
   - changing texture
- Material Mapping
   - use texture to modify material properties
   - diffuse color map or albedo color map
   - linear relationship: 
   - nonlinear relationship: roughness or bump mapping 
- Alpha Mapping
   - alpha blending or alpha testing
   - decaling
   - cutouts
   - cross tree
   - grass: three cross
   - billboarding
   - combine texture animation and alpha maps
      - plant growth, explosion, atmospheric effects
   - antialiasing the object edges
   - alpha testing
      - problem with mipmap
         - too transparent
      - black edge
- Bump Mapping
   - modify normal direction
   - can compute tangent-space basis on the fly in the shader
   - Blinn's Method
      - bu, bv
      - height field
         - use height field to get bu, bv
   - Normal Mapping
      - filtering normal map is hard
- Parallex Mapping
   - Parallex Occlusion Map or relief map
   - height field 代表的是一个曲面，当视角不同的时候看到的是不同的。
      - height field tracing
- Textured Light


#  Shadows
- Introduction
   - occluders
   - receivers
   - light source
   - shadow=umbra + penumbra
   - hard shadow: only umbra
   - soft shadow
- Planar Shadow
   - Projection Shadow
      - l, v, p, n, d
      - Mv=p
      - 运用投影矩阵到物体上，然后用暗色无光照着色。要做适当偏移，以免跟地面出现zigzag。
      - 更安全的做法：先画地面，再画投影阴影（关闭z buffer），再画其他物体。
      - 如果地面是个矩形区域，画阴影超出矩形区域，则使用stencil buffer。先把矩形地面画到屏幕和stencil buffer，然后把z buffer关掉，用stencil buffer去控制绘制投影阴影在该画的区域，最后再画其他物体。
      - 另一个技术是把阴影画到纹理上，然后用纹理去调整曲面亮度。缺点是用纹理的话常常会精度不够，一个texel cover多个pixel（magnified）。
   - Soft Shadow
      - 面积光 area light
      - 绘制方法1：采样面积光得到多个点光源，投影阴影，叠加平均。这种方式很慢。
      - 绘制方法2：采样面积光得到多个点光源，n pass绘制到texture，叠加平均。同样很慢。可以用来生成对比其他更快算法的baseline。
      - convolution or filtering
         - 得到半透明的阴影纹理
         - 与现实世界复合
         - 缺点是在物体与地面接触的地方看起来会不自然
- Shadows on Curved Surface
   - projective texture
   - 用uv来定位位置
   - drop shadow
   - shadow to mipmap
   - 
      
      

# Light Map

- Reference 
   - [https://github.com/TheRealMJP/BakingLab](https://github.com/TheRealMJP/BakingLab)
   - [https://github.com/ands/lightmapper](https://github.com/ands/lightmapper)
   
- 🦐BB

# Irradiance Volume

- Reference
[https://github.com/pyalot/webgl-deferred-irradiance-volumes](https://github.com/pyalot/webgl-deferred-irradiance-volumes)
