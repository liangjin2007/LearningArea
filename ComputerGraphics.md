# 渲染
### PBRT架构
### Surface Material(bxdfs)
### Hair, Fur, Feathers
Maya-Geo To Maya Hair 2, GMH2 for short, https://www.youtube.com/watch?v=Rtzfkig9-PY
Hair and water interaction http://www.cs.columbia.edu/cg/liquidhair/, https://github.com/nepluno/libWetHair

### Subsurface scattering， 皮肤建模
Diffusion models

### Volumes
just attenuation
single scattering
multiple scattering
Emission
Nested/overlapping volumes
Volumes with motion blur

### 光模拟算法 Light Simulation Algorithm
Integrators:
单向路径追踪uni-directional path tracing
双向路径追踪bi-directional path tracing
VCM
UPBP
specular rays
diffuse rays
camera rays

### Vertex connection and merging: VCM


##  [[cs348b]Image Synthesis Techniques](http://graphics.stanford.edu/courses/cs348b/)


# 已读论文列表
- [2003]Bilateral Mesh Denoising 
  - 两套权重相乘wc, ws
    - wc：与距离相关的smoothing权重是gaussian filter wc=exp(-x^2/sigma_c^2)
    - ws：与像素相关的特征保留权重或者称为相似性权函数是 ws=exp(-x^2/sigma_s^2)
    - https://github.com/liangjin2007/data_liangjin/blob/master/bilateral_filtering.jpg?raw=true
- [2004]A Remeshing Approach to Multiresolution Modeling
  - laplace operator离散化
  - Euler-Lagrange PDE算子线性离散化
  - multigrid solver
  - direct solver
  - iterative solver 
- [2002]Spanning Tree Seams for Reducing Parameterization Distortion of Triangulated Surfaces
  - Dijkstra shortest path algorithm
  - Minimal Spanning Tree algorithm
  
- [2004]MESH SMOOTHING SCHEMES BASED ON OPTIMAL DELAUNAY TRIANGULATIONS
  - laplace smoothing
  - OPTIMAL DELAUNAY TRIANGULATION Smoother
  - Centroid Voronoi Tesselation Smoother
  - 此论文一堆公式，最终得出的是类似于laplace smoothing一样的算子，唯一的不同是权重。
  
- [2002]Least Squares Conformal Maps for Automatic Texture Atlas Generation
  - 共形映射：把圆映射到圆
  - 如何把问题进行转化
  
- [2002]Surface Simplification Using Quadric Error Metrics
  - 优先队列
  - edge swap
  - edge clapse
