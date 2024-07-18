# 目录
- 3D deep learning
- A Comprehensive Survey on 3D Content Generation

  
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


- [2019][CVPR][3200+引用]DeepSDF， facebookresearch
  - 背景
  ```
    1. 3D Shape learning的表示：
      1.1. 基于点的，比如雷达数据，至少5篇以上工作
      1.2. 基于网格的，至少8篇以上的工作
      1.3. 基于体素的，至少6篇以上的工作
  
    2. Representation Learning技术：
      GAN, Auto Encoder, Auto Decoder(Optimize Latent Vectors),
    3. Shape Completion
      RBF拟合, Poisson reconstruction, data-driven method, 
  ```
  
  - DeepSDF ![](https://github.com/liangjin2007/data_liangjin/blob/master/deepsdf.png?raw=true)
  ```
  网络要能够输入query point p, 返回sdf值
  
  一个target mesh对应训练一个nn，这肯定不太实用
  
  学习一系列Shapes的latent space
  
  Conditioned latent vector. (z, x) -> (sdf)
  
  代码 https://github.com/facebookresearch/DeepSDF.git
  
  ```
- [2024][sig]Neural Geometry Fields For Meshes 
```
无代码

Geometry Images方法的neural network版本， 优点是不用做texture packing等。

Quad Patch, bilinear interpolation

使用了positional encoding in Nerf, enc(v, f) = (enc(v), f) = (sin(2^0 v), cos(2^0 v), ..., sin(2^L v), cos(2^L v), f)

Jittering

直接优化|sdf_pred - sdf_gt|不够稳定， 相反通过inverse rendering优化appearance更稳定。
```






### Single Image 3D Reconstruction
- [2017][cvpr][stanford]A point set generation network for 3D object reconstruction from a single image
![xxx](https://github.com/liangjin2007/data_liangjin/blob/master/PointSetGenerationNetwork.png)
![xxx](https://github.com/liangjin2007/data_liangjin/blob/master/VAE1.png)

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



- [2019]Learning Implicit Fields for Generative Shape Modeling
```
代码地址 https://github.com/czq142857/implicit-decoder
```

### [Polygen](https://www.youtube.com/watch?v=XCrjpIRkVCU&list=PLA9fKWDqz_2LZgK32sHq5q13T4O1SXubZ&index=10)


### A Survey on 3D Human Avatar Modeling: From Reconstruction to Generation

## [2024]A Comprehensive Survey on 3D Content Generation
- 3D Native Generation Models。   分成Object，Scene，Human三个类别来讲
```
Object： 训练数据集{3D对象， 对应的文本标注captions}，
  [2019]Text2Shape 几万数据集
  [2022]Text2Shape++ or ShapeCraft 几十万数据集
  [2023]SDFusion 允许多模态输入条件
  以上三个限制于训练数据集的大小，只能处理有限的对象类别。为了处理大的large-vocabulary 3D generation：
  [2022]Point-E ：  image-to-point diffusion model
  [2023]Shape-E ：   重建SDF
  这两个是几百万数据集，但是数据集没有发布出来。
  Objaverse数据集，相对于上面两篇的数据集更小一些
  [2024]LRM: imageto-triplane latent space, and triplane-based implicit representation
  [2024]DMV3D: T-step diffusion model
  [2023]TextField3D:
```

```
Scene
  
```
```
Human
  [2003]3DMM
  [2015]SMPL: 高细节几何比如头发和衣服的建模方面有局限。
  [2019]PIFU: 从3D扫描或者multiple view images学习。从单图学习pixel-aligned implicit function.
  [2022]HeadNeRF
  [2021]SMPLicit
  [2022]gDNA
  [2023]Rodin: tri-plane 表示
```
- 2D prior-based 3D generative model
```
Object
  [2023]DreamFusion: 用预训练的2D diffusion model指导per-text或者per-image的3D生成。multi-face Janus problem
  [2023]Magic3D:  coarse-to-fine optimization strategy with two stages
  [2023]Fantasia3D: disentangled geometry and appearance modeling, advancing text-to-3D quality.spatially-varying bidirectional reflectance distribution functions to learn surface materials for photorealistic rendering of generated geometry
  [2023]ProlificDreamer: variational score distillation方法
  [2023]DreamControl: adaptive viewpoint sampling and boundary integrity metrics.
  [2024]DreamGaussian: DreamGaussian introduced an efficient algorithm to convert the resulting Gaussians into textured meshes
  生成纹理
  [2023]TEXTure
  [2023]TexFusion: 利用预训练的depth-to-image模型？
```
```
Scene
  [2023]Text2room: 大模型+inpainting
  [2023]LucidDreamer
  [2023]SceneTex
  [2023]SceneDreamer
  [2024]CityDreamer
```
```
Human Avatar
  [2022]AvatarCLIP
  [2023]HeadSculpt: SDS
  [2023]DreamWaltz
  [2023]DreamHuman
  [2023]HumanGaussian
```
- Hybrid 3D Generative Method
```
While early 3D native generative methods were limited by scarce 3D datasets, and 2D prior methods could only distill limited 3D geometric knowledge, researchers explored injecting 3D information into pretrained 2D models. Emerging approaches included fine-tuning Stable Diffusion on multi-view object images to generate consistent perspectives, as well as 3D reconstruction and generation from multiple views.
```
```
Object
  [2023]Zero123: applies 3D data to fine-tune pre-trained 2D diffusion models
  [2023]One-2-3-45: cross-view attention + multiple-view conditioned 3D diffusion + coarse-to-fine textured mesh prediction
  [2024]SyncDreamer
  [2024]MVDream
  [2023]Wonder3D: normal modal fine-tune multiple view stable diffusion model to concurrently output RGB and normal maps.
  [2023]UniDream: use diffusion prior to generate multiple view albedo-normal information
  [2023]Dreamcraft3d: DMTet表示， Score distillation sampling
  [2023]Gsgen: 3D Gaussian
  [2023]Instant3d
```
```
Scene
  [2023]MVDiffusion
  [2023]ControlRoom3D
  [2023]SceneWiz3D
```
```
HumanAvatar
  DreamFace
```
```
Dynamic
  [2023]MAV3D: hex-plane
  [2023]Animate124
  [2023]4D-fy
  [2023]4DGen
  [2023]DreamGaussian4D
```

- 局限和TODO
```
几何上：不能建模compact mesh以及合理的布线。
纹理上：不能产生高细节的纹理映射，也难以消除灯光和阴影。
材质属性：不能很好地支持。
精确控制上： text/image/sketch based 方法不能精确输出3D资产来满足条件需求。
编辑能力： 编辑能力也不太够。
速度上： 相比较于基于NERF的优化方法，feed-forward和基于GS的SDS方法要快一些，但是也伴随着低质量。

总的来说，要生成产品级质量，scale和精度的3D内容，仍然没有解决。
```

- 数据
```
在收集百万级的3D对象上面临着挑战。
这可能可以通过创建一个3D开放世界游戏平台来让用户上传自定义的模型。
另外通过从multiple view images提取丰富的隐式3D知识。
无监督。
自监督。
```

- 模型
```

```






