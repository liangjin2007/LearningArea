# Latex公式编辑
- 在线公式编辑器 https://www.codecogs.com/latex/eqneditor.php
- Latex语法 

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
  

