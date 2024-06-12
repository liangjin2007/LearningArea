# 目录
- 3D deep learning

## 3D deep learning

### min距离的可微化
```
Chamfer distance 是一种常用于评估两组点云之间相似度的损失函数，经常在深度学习中的3D形状生成和重建任务中使用。Chamfer distance 可微的原因在于其定义方式。
Chamfer distance 由两部分组成，分别对应于两个点云之间的最小距离和最大距离：
对于点云A中的每一个点，找到点云B中最近的点的距离的平均值。
对于点云B中的每一个点，找到点云A中最近的点的距离的平均值。
形式化地，假设我们有两个点云 ( X = {x_1, x_2, ..., x_n} ) 和 ( Y = {y_1, y_2, ..., y_m} )，则 Chamfer distance ( D_C ) 定义为：
[ D_C(X, Y) = \frac{1}{n} \sum_{x \in X} \min_{y \in Y} ||x - y||^2 + \frac{1}{m} \sum_{y \in Y} \min_{x \in X} ||y - x||^2 ]
Chamfer distance 是可微的，原因如下：
最小距离的可微性：最小距离操作本质上是一个求取两个集合中点对之间距离最小值的操作。虽然最小值函数本身不可微（因为当多个点具有相同的最小距离时，导数未定义），但是我们可以通过使用所谓的“soft”最小化技术来近似它，例如使用softmax函数。这种方法可以赋予最小距离操作连续的梯度，从而使其可微。
求和的可微性：求和操作是可微的，因为它是连续的，并且对于每个单独的项都有明确的梯度。
平方欧几里得距离的可微性：平方欧几里得距离 ( ||x - y||^2 ) 是可微的，因为它是一个关于其输入的二次函数。
综上所述，通过使用可微的近似最小化技术和可微的平方欧几里得距离，Chamfer distance 整体上变得可微。这使得它可以在基于梯度的优化算法（如梯度下降）中使用，从而在深度学习模型中用于端到端的训练。


在 Chamfer Distance 中，使用加权距离的数学表示涉及到将 softmax 函数的输出（即概率分布）用于加权点云A中的点 ( x ) 到点云B中所有点的距离。以下是具体的数学步骤：
对于点云A中的每个点 ( x )，计算它到点云B中每个点 ( y ) 的欧几里得距离 ( d(x, y) )。
应用 softmax 函数来获得权重 ( P(y|x) )，这个权重表示点 ( y ) 相对于点 ( x ) 的其他所有邻近点的相对概率：
[ P(y|x) = \frac{e^{-d(x, y)}}{\sum_{y' \in Y} e^{-d(x, y')}} ]
其中 ( Y ) 是点云B中的所有点。
使用这些权重来计算加权距离，即点 ( x ) 到点云B的加权平均距离：
[ D(x) = \sum_{y \in Y} P(y|x) \cdot d(x, y) ]
这个式子表示点 ( x ) 到点云B中所有点的加权距离。
对点云A中的所有点 ( x ) 进行平均，得到点云A到点云B的加权距离的平均值：
[ D_A = \frac{1}{n} \sum_{x \in X} D(x) ]
同样的步骤也适用于点云B到点云A的加权距离：
[ D_B = \frac{1}{m} \sum_{y \in Y} \sum_{x \in X} P(x|y) \cdot d(y, x) ]
其中 ( P(x|y) ) 是点 ( y ) 到点云A中所有点的加权概率。
最终的 Chamfer Distance ( D_C ) 是这两个加权距离的平均值：
[ D_C = D_A + D_B ]
通过这种方式，我们得到了一个连续的、可微的 Chamfer Distance 损失函数，可以用于深度学习模型的训练。这种 soft 最小化技术使得原始的 Chamfer Distance 在最小化过程中变得平滑，从而允许使用梯度下降等优化算法进行端到端的训练。
```

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
$$
d_{EMD}(S_1, S_2) = \min_{\phi:S_1 \to S_2} \sum_{x \in S_1}||x - \phi(x)||_2
$$

$$d_{CH}(S_1, S_2) = \sum_{x \in S_1}\min_{y \in S_2}||x-y||_2^2$$

$$+$$

$$\sum_{y \in S_2}\min_{x \in S_1}||x-y||_2^2$$

### Single Image 3D Reconstruction
- [2017][cvpr][stanford]A point set generation network for 3D object reconstruction from a single image
```
conditional sampler
  ground truth数据: {图片，对应点云}
  
  encoder stage and predictor stage
    encoder stage:
      输入{Image_i, random vector r_i}, 输出latent vector(embedding space)
    predictor stage:
      输入 latent vector, 输出N x 3 矩阵M。
  
  We train a neural network G as a conditional sampler
  from P(·|I):
    S = G(I, r; Θ)
  where Θ denotes network parameter, r ∼ N(0, I) is a
  random variable to perturb the input 1
  . During test time
  multiple samples of r could be used to generate different
  predictions.

loss:
  Chamfer Distance(CD)： 是可微的。
  Earth Mover's Distance(EMD) ： EMD是个优化问题 assignment问题。

loss是否可微可以从极限定义的角度考虑， 比如此论文中说，考虑极小的位置挠动，EMD的最优双射\phi是唯一的。因此EMD是几乎处处可微。
每个EMD计算太昂贵，1+epsilon近似且可在GPU上实现。([1985]A distributed asynchronous relaxation algorithm for the assignment problem)。

对形状空间的理解：
Shape space Despite remarkable expressive power embedded in the deep layers, neural networks inevitably encounter uncertainty in predicting the precise geometry of an object. Such uncertainty could arise from limited network capacity, insufficient use of input resolution, or the ambiguity of groundtruth due to information loss in 3D-2D projection. Facing the inherent inability to resolve the shape precisely, neural networks tend to predict a “mean” shape averaging out the space of uncertainty. The mean shape carries the characteristics of the distance itself.

Min-of-N loss (MoN)：


```

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
