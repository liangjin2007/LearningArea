# 方法
#### PBD
[2017]A Survey on Position Based Dynamics, 2017
- 目标求dp使得C(p + dp) = 0
- C(p)定义时应该要跟平移和旋转不相关。
- 动量守恒![动量守恒](https://github.com/liangjin2007/data_liangjin/blob/master/simulation/momentum_conservation.png?raw=true)
- dp取C(p)梯度方向可以保证动量守恒
- pbd约束投影，非线性牛顿solver（即一阶泰勒展开） ![pbd约束投影](https://github.com/liangjin2007/data_liangjin/blob/master/simulation/pbd.png?raw=true)
#### XPBD
思路 C(x) -> energy potential U(x) -> 牛顿第二定律 Ma = -gradU(x) -> 隐式离散化（时间步n, 将a中心差分（n-1, n, n+1时间步）， 右侧x用n+1时间步
- 定义grad为行向量
- 起始 ![起始](https://github.com/liangjin2007/data_liangjin/blob/master/simulation/xpbd_1.png?raw=true)
- 离散化过程![离散化过程](https://github.com/liangjin2007/data_liangjin/blob/master/simulation/xpbd_2.png?raw=true)
- damping![damping](https://github.com/liangjin2007/data_liangjin/blob/master/simulation/xpbd_3.png?raw=true)
- 算法![算法](https://github.com/liangjin2007/data_liangjin/blob/master/simulation/xpbd_4.png?raw=true)
- 例子：spring, chain, cantilever beam, cloth, balloon
#### PD
#### FEM
- [SIGGRAPH 2012 Course](http://www.femdefo.org/)
#### SPH
- https://interactivecomputergraphics.github.io/SPH-Tutorial/pdf/SPH_Tutorial.pdf
#### MPM
- MPM论文列表 https://www.seas.upenn.edu/~cffjiang/mpm.html
- physics based animation http://www.physicsbasedanimation.com/resources-courses/
#### force-based method

# 各种模拟
#### hair
[ue4 real time hair advances](https://www.fxguide.com/fxfeatured/ue4-real-time-hair-advances/)
[siggraph 2007 course Hair Modeling, Simulation and Rendering](https://hal.inria.fr/inria-00520193/document)

#### wind
- NvCloth/docs/documentation/Solver/Index.html#wind-simulation
#### thin shell
#### elastic objects
#### snow
#### lava
#### sand
#### viscoelastic fluids
#### cloth

# Taichi GAMES201
