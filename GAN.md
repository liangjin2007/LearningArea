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

## CycleGAN

## NVIDIA FUNIT

