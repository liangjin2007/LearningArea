# 目录
- 3D deep learning
## 3D deep learning
- [2019][ref2639]Occupancy Networks: Learning 3D Reconstruction in Function Space
```
Presentation Video: https://www.youtube.com/watch?v=kxKI8_Si2a0, https://www.youtube.com/watch?v=9r9TDr2Aq5A
2600+引用；如何近似三维对象的occupancy函数o : R^3 -> {0, 1}； 用神经网络学习一个泛函 x(e.g. mesh point positions) -> f(p in R^3 -> R); 转变成
学习一个Occupancy Network ： f_theta : R^3 x X -> [0, 1]。
```
$$L_B\left(\theta\right) = \frac{1}{|B|}\sum_{i=1}^{|B|}\sum_{j=1}^{K}L\left(f_{\theta}\left(p_{ij},x_i\right),o_{ij}\right)$$

- [2019]Learning Implicit Fields for Generative Shape Modeling
