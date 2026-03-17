## Pytorch
- [API文档](https://docs.pytorch.org/docs/stable/index.html)
- [Deep-Learning-With-Pytorch-Tutorials](https://github.com/dragen1860/Deep-Learning-with-PyTorch-Tutorials.git)
- [NVIDA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master) 进阶学习

## 概率与惊喜 https://zhuanlan.zhihu.com/p/573385147
```
1.概率
p(x) ->
惊喜度 1/p(x)， 运算性质不好 ->
log(1/(p(x))) = -log(p(x)) -> 1.确定性事件的惊喜度=0，2.如果有多个独立事件同时发生，他们产生的惊喜度可以直接相加。

2.信息熵
惊喜度=香农提出的信息量
信息熵是惊喜度的期望(平均惊喜) H_p(x) = sum p(x) log(1/p(x)) or integrate p(x) log(1/p(x)) dx
对于给定均值和方差的连续分布，正态分布（高斯分布）具有最大的信息熵（也就是平均惊喜）。所以再想想为什么大量生活中会看到的随机事件分布都服从正态分布呢？说明大自然有着创造最大惊喜的倾向，或者说，就是要让你猜不透。这也是理解热力学中的熵增定律的另一个角度。

3.交叉熵
对于概率，比较经典的理解是看做是重复试验无限次后事件频率会逼近的值，是一个客观存在的值；但是贝叶斯学派提出了另一种理解方式：即将概率理解为我们主观上对事件发生的确信程度。
po
ps
H_po,ps(X) https://zhuanlan.zhihu.com/p/573385147
什么时候交叉熵等于信息熵？ 主观和客观完全匹配


4.相对熵 Kullback-Leibler Divergence，K-L 散度
如何度量主观认识和客观之间差异？当前“世界观”产生的惊喜期望和完全正确认识事件时产生的惊喜期望的差值来衡量 ->
D_KL(po||ps) = H_po,ps(X) - H_po(X) = integrate po(x) log(po(x)/ps(x)) dx ->
当我们的主观认知完全匹配客观现实的时候，KL-散度应该等于0，其它任何时候都会大于0。
KL-散度经常用于描述两个分布是否接近，也就是作为两个分布之间“距离”的度量。
不过由于运算不满足交换律，所以又不能完全等同于“距离”来理解。
机器学习中通常用交叉熵作为损失函数的原因在与，客观分布并不随参数变化，所以即使是优化KL-散度，对参数求导的时候也只有交叉熵的导数了。

5.补充
评论中有一个非常不错的问题：为什么分类问题不用MSE作为Loss？
这里面涉及到关于统计学习模型在设计上的底层原则：
1. 如果要学习/拟合的对象是一个确定（deterministic）的函数，也就是说，一个给定的x，y=f(x) 是一个确定值（只不过观测中会存在噪声），就可以且应该用mse；
2. 如果要学习/拟合的对象本身就是一个随机（stochastic）函数，也就是说，一个给定的x，y=f(x) 不存在确定值，而是存在一个分布，那么要学习也应该是一个分布，如果按照mse作为loss，学习到的很可能就只是这个随机现象的均值。所以本质的区别在于，同一个x下的不同观测值之间的波动，是要被看待为噪声，还是要被看待为想拟合的对象的固有属性。
分类问题的输入是直接观测或者特征，输出是预测值，我们可以由观测或特征可以直接推导出结果吗？一般而言不能，只能增加我们对不同结果的确信程度，因此输出是分布。
```
## MLP
## Loss
```

```
## Linear-Probe-Fine-Tuning 线性探针
## VAE
## Positional Embedding
## Transformer
## Diffusion
## DiT
## ViT




