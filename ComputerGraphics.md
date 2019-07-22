# 已读论文列表
- [2003]Bilateral Mesh Denoising 
  - 两套权重相乘wc, ws
    - wc：与距离相关的smoothing权重是gaussian filter wc=exp(-x^2/sigma_c^2)
    - ws：与像素相关的特征保留权重或者称为相似性权函数是 ws=exp(-x^2/sigma_s^2)
    - https://github.com/liangjin2007/data_liangjin/blob/master/bilateral_filtering.jpg?raw=true
- [2004]A Remeshing Approach to Multiresolution Modeling
  - **free from modeling**
  选择编辑区域，然后使用boundary constraint method
  - 历史
    - NURBS难以处理复杂的物体,导致出现了很多方法是先把NURBS离散化成三角网格，然后再在三角网格上进行处理。
    - Subdivision Mesh要求semi-regular拓扑连接关系。
    - 三角网格非常自由。
  - 多分辨率建模：
    - 把曲面分成低频的base mesn和高频的细节
    - a freeform modeling operator deforming the base surface
    - a reconstruction operator adding the detail information back onto a modified version of the base surface
