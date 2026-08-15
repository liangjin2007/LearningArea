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

## MSE & Linear Regression（Mean Squared Error） 
```
样本xi, yi
样本上的损失函数 l(yi, f(xi)) = 1/2(yi - f(xi))^2
数据集熵的经验风险(Empirical Risk)函数 L = 1/n Sum l(yi, f(xi)) = 1/n Sum 1/2(yi - f(xi))^2
几何理解：
  ->Robust Regression(鲁棒回归): RANSAC， Huber Regression
概率理解：
  MLE极大似然估计： 引入噪声概率分布epsilon, 正太分布， -log L(w, b) = -log(Multiply p(yi|xi;w,b)) = -n log(1/sqrt(2 PI sigma^2)) + 1/(2 sigma^2) sum(yi - (w xi + b))^2
```
## 0-1 Loss & Surrogate(代理) Loss  https://zhuanlan.zhihu.com/p/346935187
```
0-1 Loss: 
Logistic Loss
Hinge Loss
```
## 统计学习方法 MLE & MAP https://zhuanlan.zhihu.com/p/345024301

## Linear-Probe-Fine-Tuning 线性探针

# AE AutoEncoder
```
自编码器类似于一个非线性的PCA，是一个利用神经网络来给复杂数据降维的模型。
编码器 z = g(X)
解码器 X^ = f(z)。
我们能否把这个模型直接当做生成模型 ? -> no
因为没学z的分布导致随便采样一个z，我们并不知道哪些能够生成有用的图片。
```
## VAE 
- https://zhuanlan.zhihu.com/p/348498294
- https://zhuanlan.zhihu.com/p/34998569
架构基本一致（都是 encoder-decoder），区别不在网络层数/拓扑，而在于：

Latent 层输出不同：AE 输出确定性向量 $z$；VAE 输出分布参数 $\mu$ 和 $\log\sigma^2$（两个 head），再经重参数化 

$z=\mu+\sigma\varepsilon$ 采样。同一 latent 维度下 VAE 该层参数量约为 AE 的 2 倍。

训练目标不同：AE 只有重建损失；VAE 额外加 KL 散度项把 latent 推向 $\mathcal{N}(0,I)$，即 $\mathcal{L}=\text{重建}+\beta\cdot KL$（β-VAE 形式）。

性质差异：AE 学到确定性压缩表示（非生成式）；VAE latent 平滑连续，可任意采样 $z\sim\mathcal{N}(0,I)$ 生成新样本，是真正的生成模型。

每个输入X对应编码/输出为一个分布。

```
在Autoencoder的基础上，显性的对z的分布p（z）进行建模，使得自编码器成为一个合格的生成模型，我们就得到了Variational Autoencoders。


p(X) = Sum p(X|Z) p(Z)

Loss
  reconstruction_function = nn.BCELoss(size_average=False)  # mse loss
  def loss_function(recon_x, x, mu, logvar):
      """
      recon_x: generating images
      x: origin images
      mu: latent mean
      logvar: latent log variance
      """
      BCE = reconstruction_function(recon_x, x)
      # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
      KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
      KLD = torch.sum(KLD_element).mul_(-0.5)
      # KL divergence
      return BCE + KLD
```
## Conditioned VAE https://zhuanlan.zhihu.com/p/88750084
- ![添加是怎么加的](https://pic2.zhimg.com/v2-85fff6a27c1f96a2d0b675243319407b_1440w.jpg)
- 样例代码
```
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class CVAE(nn.Module):
    """Conditional Variational AutoEncoder for MNIST.

    Condition c is the digit label (one-hot, 10-dim), concatenated into both
    the encoder's first FC layer and the decoder's input.
    """

    def __init__(self, latent_dim=32, n_classes=10, in_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(256 + n_classes, latent_dim)
        self.fc_logvar = nn.Linear(256 + n_classes, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 256),
            nn.ReLU(True),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, c):
        h = self.encoder(x)
        h = torch.cat([h, c], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc), mu, logvar

    def sample(self, n_per_class, device="cpu"):
        """Generate n_per_class images for each of the n_classes digits."""
        self.eval()
        grid_rows = []
        for label in range(self.n_classes):
            c = F.one_hot(torch.tensor(label), self.n_classes).float().unsqueeze(0)
            c = c.repeat(n_per_class, 1).to(device)
            z = torch.randn(n_per_class, self.latent_dim).to(device)
            with torch.no_grad():
                imgs = self.decoder(torch.cat([z, c], dim=1))
            grid_rows.append(imgs)
        return torch.cat(grid_rows, dim=0)


def cvae_loss(x_recon, x, mu, logvar):
    """CVAE loss (ELBO negative), reduction='sum'.

    recon = -E[log p(x|z)] via BCE  (MNIST pixels in [0, 1])
    kl    = KL(q(z|x) || N(0, I))   summed over dims/batch
    """
    recon = F.binary_cross_entropy(x_recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon, kl


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs, batch_size, latent_dim = 10, 128, 32
    sample_dir = "samples"
    os.makedirs(sample_dir, exist_ok=True)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1, 1, 28, 28))]
    )
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=tf),
        batch_size=batch_size,
        shuffle=True,
    )

    model = CVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        total_recon, total_kl = 0.0, 0.0
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            c = F.one_hot(y, model.n_classes).float().to(device)
            x_recon, mu, logvar = model(x, c)
            loss, recon, kl = cvae_loss(x_recon, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_recon += recon.item()
            total_kl += kl.item()

        n = len(train_loader.dataset)
        print(
            f"epoch {epoch:02d} | recon={total_recon / n:.3f} | "
            f"kl={total_kl / n:.3f} | total={total_recon + total_kl:.2f}"
        )

        samples = model.sample(n_per_class=8, device=device)
        save_image(
            samples,
            f"{sample_dir}/cvae_samples_epoch{epoch:02d}.png",
            nrow=8,
            padding=2,
        )


if __name__ == "__main__":
    main()
```
```
样例代码中
h = encoder(x)
fc_mu = mlp1(h) # mlp1为 latent_dim -> 256 + n_classes
fc_logvar = mlp2(h) # mlp1为 latent_dim -> 256 + n_classes
decoder(fc_mu, fc_logvar)
```
## TRPO
```
https://zhuanlan.zhihu.com/p/26308073
```
## Positional Embedding
## Transformer https://zhuanlan.zhihu.com/p/525106459

## Diffusion
## DiT
## ViT




