# 已读论文列表
- [2003]Bilateral Mesh Denoising 
  - 两套权重相乘wc, ws
    - wc：与距离相关的smoothing权重是gaussian filter wc=exp(-x^2/sigma_c^2)
    - ws：与像素相关的特征保留权重或者称为相似性权函数是 ws=exp(-x^2/sigma_s^2)
    - https://github.com/liangjin2007/data_liangjin/blob/master/bilateral_filtering.jpg?raw=true
- [2004]A Remeshing Approach to Multiresolution Modeling
  - laplace operator离散化
  - Euler-Lagrange PDE算子线性离散化
  
- [2002]Spanning Tree Seams for Reducing Parameterization Distortion of Triangulated Surfaces
  - Dijkstra shortest path algorithm
  - Minimal Spanning Tree algorithm
  
- [2004]MESH SMOOTHING SCHEMES BASED ON OPTIMAL DELAUNAY TRIANGULATIONS
  - laplace smoothing
  - OPTIMAL DELAUNAY TRIANGULATION Smoother
  - Centroid Voronoi Tesselation Smoother

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align}\notag&space;\dot{x}&=\mathbf{A}x&plus;\mathbf{B}u\\&space;y&=\begin{bmatrix}1&0\\&space;0&1\end{bmatrix}x&plus;\begin{bmatrix}1&0\\&space;0&1\end{bmatrix}u&space;\end{align}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align}\notag&space;\dot{x}&=\mathbf{A}x&plus;\mathbf{B}u\\&space;y&=\begin{bmatrix}1&0\\&space;0&1\end{bmatrix}x&plus;\begin{bmatrix}1&0\\&space;0&1\end{bmatrix}u&space;\end{align}" title="\begin{align}\notag \dot{x}&=\mathbf{A}x+\mathbf{B}u\\ y&=\begin{bmatrix}1&0\\ 0&1\end{bmatrix}x+\begin{bmatrix}1&0\\ 0&1\end{bmatrix}u \end{align}" /></a>
