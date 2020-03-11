# GAN

## 一般概念
- [简书](https://www.jianshu.com/p/b8c81f343c30)
- 输入：真实图片x及一个随机的噪声z
- 输出：G与D
  - 其中D(x)返回一个[0,1]的浮点数表示真实图片x的概率。
  - 其中G(z)表示用z生成大量的图片去骗过D, 所以用的是随机变量。
- 优化目标： max_D, min_G。 
  - max_D用来使得E[log(D(x))]+E[log(1-D(G(z)))];
  - min_G其实是min E[log(1-D(G(z)))]
  
## DCGAN
```
Deep Convolutional Generative Adversarial Networks
1. 取消所有pooling层。G网络中使用转置卷积（transposed convolutional layer）进行上采样，D网络中用加入stride的卷积代替pooling。
    2. 在D和G中均使用batch normalization
    3. 去掉FC层，使网络变为全卷积网络
    4. G网络中使用ReLU作为激活函数，最后一层使用tanh
    5. D网络中使用LeakyReLU作为激活函数
```
## CycleGAN
CycleGAN的原理可以概述为：将一类图片转换成另一类图片。也就是说，现在有两个样本空间，X 和 Y，我们希望把 X 空间中的样本转换成 Y 空间中的样本。因此，实际的目标就是学习从 X 到 Y 的映射（设这个映射为 F），F 就对应着 GAN 中的生成器，F 可以将 X 中的图片 x 转换为 Y 中的图片 F(x)。对于生成的图片，我们还需要 GAN 中的判别器来判别它是否为真实图片，由此构成对抗生成网络。设这个判别器为 DY。这样的话，根据这里的生成器和判别器，我们就可以构造一个 GAN 损失，表达式为：
- 输入: 图片x及图片y
- 输出：生成网络F: x->y， 判别网络Dy
- 优化目标: max_Dy, min_F

## 条件GAN CGAN
Conditional Adversarial Networks


## NVIDIA FUNIT

