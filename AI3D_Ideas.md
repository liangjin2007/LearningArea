# 目录
- 3D deep learning
## 3D deep learning
- [2019]Occupancy Networks: Learning 3D Reconstruction in Function Space
```
2600+引用；如何近似三维对象的occupancy函数o : R^3 -> {0, 1}； 用神经网络学习一个泛函 x(e.g. mesh point positions) -> f(p in R^3 -> R); 转变成
学习一个Occupancy Network ： f_theta : R^3 x X -> [0, 1]。
```
$$L_B\left(\theta\right) \eq \frac{1}{|B|}\sum_{i=1}{|B|}\sum_{j=1}{K}L\left(f_{\theta}\left(p_{ij},x_i\right),o_{ij}\right)$$
