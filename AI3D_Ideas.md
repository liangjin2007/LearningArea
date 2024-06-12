# 目录
- 3D deep learning

## 3D deep learning



### Encoder Network
- [2017]Learning Representations and Generative Models for 3D Point Clouds
```
The first deep generative models for point clouds（给一个z能输出一个新的点云）
AE
GAN
permutation-invariant metrics ：
  Earth Mover’s distance (EMD)
  Chamfer Distance(CD)
```
$$d_{EMD}(S_1, S_2) = \min_{\phi:S_1 \to S_2} \sum_{x \in S_1}||x - \phi(x)||_2$$

$$ d_{CH}(S_1, S_2) = \sum_{x \in S_1}\min_{y \in S_2}||x-y||_2^2 + \sum_{y \in S_2}\min_{x \in S_1}||x-y||_2^2 $$

### Single Image 3D Reconstruction



- [2019][ref2639]Occupancy Networks: Learning 3D Reconstruction in Function Space
```
Presentation Video:
  https://www.youtube.com/watch?v=9r9TDr2Aq5A

一种基于学习的3D重建的高效、低内存的高分辨率任意拓扑表示。之前的方法的缺陷： coarse 3D geometry或者表示的空间范围有限。

这篇方法隐式表示3D几何为连续decision boundary of a deep neural network classifier。无限分辨率。一种新的encoder。能encode各种各样的输入。

存在的方法： voxel-based表示， point-based表示， mesh表示。
AE, GAN, GMM, 



2600+引用；如何近似三维对象的occupancy函数o : R^3 -> {0, 1}； 用神经网络学习一个泛函 x(e.g. mesh point positions) -> f(p in R^3 -> R); 转变成
学习一个Occupancy Network ： f_theta : R^3 x X -> [0, 1]。
```
$$L_B\left(\theta\right) = \frac{1}{|B|}\sum_{i=1}^{|B|}\sum_{j=1}^{K}L\left(f_{\theta}\left(p_{ij},x_i\right),o_{ij}\right)$$



- [2019]Learning Implicit Fields for Generative Shape Modeling
```
代码地址 https://github.com/czq142857/implicit-decoder
```
