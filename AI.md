## Pytorch
- [API文档](https://docs.pytorch.org/docs/stable/index.html)
- [Deep-Learning-With-Pytorch-Tutorials](https://github.com/dragen1860/Deep-Learning-with-PyTorch-Tutorials.git)
- [NVIDA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master) 进阶学习

## 目录
- [1.概率与惊喜](#1.概率与惊喜)
- [2.Likelihood](#2.Likelihood)
- [3.Divergence(散度)](#3.Divergence(散度))
- [4.MSE](#4.MSE)
- [5.Loss](#5.Loss)
- [6.统计学习方法](#6.统计学习方法)
- [7.Linear-Probe-Fine-Tuning](#7.Linear-Probe-Fine-Tuning)
- [8.AutoEncoder](#8.AutoEncoder)
- [9.VAE](#9.VAE)
- [10.CVAE](#10.CVAE)
- [11.重参数化](#11.重参数化)
- [12.GAN](#12.GAN)
- [13.PositionalEmbedding](#13.PositionalEmbedding)
- [14.DiffusionModel](#14.DiffusionModel) 
- [15.NormalizedFlow](#15.NormalizedFlow) 
- [16.Transformer](#16.Transformer)
- [17.Flow Matching](#17.Flow Matching)
- [18.DiT](#18.DiT)
- [19.ViT](#19.ViT)

## 1.概率与惊喜 https://zhuanlan.zhihu.com/p/573385147
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
## 2.Likelihood
```
Likelihood（似然）
给定一组已观测数据 X，likelihood 是在某个参数 θ 下，观测到这些数据的概率（或概率密度值）：

L(θ) = P(X | θ)   # 数据 X 固定，θ 可变
关键区分：

Probability：P(X|θ)，θ 固定、X 可变 —— 描述"结果"
Likelihood：P(X|θ)，X 固定、θ 可变 —— 描述"参数的好坏"
最大似然估计 (MLE)：找使 L(θ) 最大的 θ。实践中常用 log-likelihood（log L(θ)），因为连乘变连加、数值稳定。例：给定 100 次抛硬币结果，求正面的 θ，MLE = 正面次数/100。

在 VAE 里就是 log p(x|z)（重建项）——衡量给定 latent z 能多大程度生成出 x。
```

## 3.Divergence(散度)
- Minimize KL Divergence
```
Divergence（散度）
散度是**衡量两个概率分布 P 和 Q 有多"不像"**的度量（非对称距离）。最常用：

KL Divergence（Kullback-Leibler）：

KL(P || Q) = Σ P(x) · log[P(x) / Q(x)]     （离散）
KL(P || Q) = ∫ P(x) · log[P(x) / Q(x)] dx   （连续）
性质：

非对称：KL(P||Q) ≠ KL(Q||P)，方向有语义——"用 Q 近似 P 的代价"
≥ 0，等于 0 当且仅当 P = Q
非度量（不满足对称性、三角不等式），所以叫"散度"而非"距离"
其他散度：JS 散度（对称化 KL，GAN 早期用它）、Wasserstein 距离（WGAN 用）、f-divergence 家族（统一形式 Σ Q·f(P/Q)）。

两者的联系（在 VAE 中的体现）
VAE 的 ELBO loss 正是二者的组合：

ELBO = E_q[log p(x|z)]    ← likelihood（重建项）
       - KL(q(z|x) || p(z))  ← divergence（正则项，拉近后验与先验）
直觉：likelihood 要求"生成得越像越好"，KL 要求"latent 分布别偏离先验太多"。二者博弈 → 平衡生成质量与 latent 空间平滑性。

一句话：likelihood 是"数据固定、参数可变时的出现概率"；divergence 是"两个分布之间的距离度量"；VAE 用 likelihood 保证重建、用 KL 散度保证 latent 正则。
```



## 4.MSE
```
样本xi, yi
样本上的损失函数 l(yi, f(xi)) = 1/2(yi - f(xi))^2
数据集熵的经验风险(Empirical Risk)函数 L = 1/n Sum l(yi, f(xi)) = 1/n Sum 1/2(yi - f(xi))^2
几何理解：
  ->Robust Regression(鲁棒回归): RANSAC， Huber Regression
概率理解：
  MLE极大似然估计： 引入噪声概率分布epsilon, 正太分布， -log L(w, b) = -log(Multiply p(yi|xi;w,b)) = -n log(1/sqrt(2 PI sigma^2)) + 1/(2 sigma^2) sum(yi - (w xi + b))^2
```
## 5.Loss 
- https://zhuanlan.zhihu.com/p/346935187
```
0-1 Loss: 
Logistic Loss
Hinge Loss
```
## 6.统计学习方法 
- MLE & MAP https://zhuanlan.zhihu.com/p/345024301

## 7.Linear-Probe-Fine-Tuning
线性探针

# 8.AutoEncoder
```
自编码器类似于一个非线性的PCA，是一个利用神经网络来给复杂数据降维的模型。
编码器 z = g(X)
解码器 X^ = f(z)。
我们能否把这个模型直接当做生成模型 ? -> no
因为没学z的分布导致随便采样一个z，我们并不知道哪些能够生成有用的图片。
```
## 9.VAE 
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
我看之前你给出的cvae_mnist.py中的如下代码
def cvae_loss(x_recon, x, mu, logvar):    """CVAE loss (ELBO negative), reduction='sum'.


    recon = -E[log p(x|z)] via BCE  (MNIST pixels in [0, 1])
    kl    = KL(q(z|x) || N(0, I))   summed over dims/batch
    """
    recon = F.binary_cross_entropy(x_recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon, kl

请问为什么E_q[log p(x|z)]就是F.binary_cross_entropy(x_recon, x, reduction="sum")，帮我推导一下
这是纯推导问题，直接回答（不涉及文件改动）。

结论先给
recon = BCE(x_recon, x, reduction='sum') 并不等于 E_q[log p(x|z)]，而是它的相反数：

recon = -E_q[log p(x|z)]
所以 cvae_loss 返回的 recon + kl 是负 ELBO（对应 docstring 里的 "ELBO negative"），最小化它 = 最大化 ELBO。

推导
第 1 步：定义似然模型。 MNIST 像素 ∈ [0,1]，decoder 输出 x_recon（Sigmoid 后 ∈ (0,1)），把它解释为每个像素独立的 Bernoulli 概率：

p_θ(x | z) = Π_i p_i^{x_i} · (1 - p_i)^{1-x_i},    p_i = x_recon_i
第 2 步：取对数，拆开连乘：

log p_θ(x | z) = Σ_i [ x_i·log p_i + (1 - x_i)·log(1 - p_i) ]      …(1)
第 3 步：期望的蒙特卡洛估计。 E_q(z|x)[log p(x|z)] 无法解析计算，用一次采样近似（VAE 标准做法，每个样本重参数化采一个 z）：

E_q[log p(x|z)] ≈ log p_θ(x | z^*)      （z^* 为一次采样）
第 4 步：看 BCE 的展开式。 PyTorch 的 F.binary_cross_entropy 对单样本、reduction='sum' 时是：

BCE = -Σ_i [ x_i·log p_i + (1-x_i)·log(1-p_i) ]      …(2)
对比 (1)(2)：

BCE = -log p_θ(x | z)   ⇒   log p_θ(x|z) = -BCE
对 batch 求和（sum 累积所有样本和像素）后两边同取期望：

E_q[log p(x|z)] = -BCE_sum        ⇒   recon = -E_q[log p(x|z)]   ∎
最后拼上 KL
loss = recon + kl
     = -E_q[log p(x|z)] + KL(q(z|x) || p(z))
     = -ELBO
两个值得注意的点
减号是"对数似然"和"损失"的方向差异：likelihood 要最大化，loss 要最小化，所以 recon 实际是负对数似然（NLL），写成"损失"形式。
隐含假设：把像素当成独立 Bernoulli（0/1 二值概率），忽略了像素的连续强度信息——这是 MNIST 场景的常见近似。若改用高斯似然（p(x|z) = N(x; x_recon, I)），推导可得 recon 对应 MSE：-log p(x|z) ∝ ||x - x_recon||²。
```
## 10.CVAE
- https://zhuanlan.zhihu.com/p/88750084
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
## 11.重参数化 
- https://zhuanlan.zhihu.com/p/561328468
针对离散选择问题。

```
（1）使用argmax函数不可导；

（2）单纯采用argmax函数失去了采样的意义，缺乏探索性。
```
- 连续高斯重参数化技巧（Gaussian reparameterization trick）
```
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps
```
- Gumbel-Softmax
```
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class CategoricalCVAE(nn.Module):
    """Conditional VAE with a discrete (categorical) latent, trained via
    Gumbel-Softmax reparameterization (Jang et al. 2017 / Maddison et al. 2017).

    The latent is a single categorical variable with `n_categories` classes.
                          
                                                                 
                                                                                
                                                                        
                                                                             
                                                                                         

    Condition c is the digit label (one-hot, 10-dim), concatenated into both the
    encoder's first FC layer and the decoder's input.
    """

    def __init__(self, n_categories=32, n_classes=10, in_channels=1, tau=1.0):
        super().__init__()
        self.n_categories = n_categories
        self.n_classes = n_classes
        self.tau = tau

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(True),
        )
        self.fc_logits = nn.Linear(256 + n_classes, n_categories)

        self.decoder = nn.Sequential(
            nn.Linear(n_categories + n_classes, 256),
            nn.ReLU(True),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, c, tau=None, hard=True):
        """Return reconstruction, latent logits, and relaxed one-hot sample.
                                                                           
                                                                                  
                                                          
                                                                            
                                                

        hard=True  -> straight-through estimator (forward one-hot, backward soft)
        hard=False -> fully relaxed (soft) sample used for decoding
        """
        tau = self.tau if tau is None else tau
        h = self.encoder(x)
        h = torch.cat([h, c], dim=1)
        logits = self.fc_logits(h)

        # Gumbel-Softmax reparameterization: differentiable sample from a
        # categorical distribution q(z|x) = softmax(logits)
        z = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc), logits, z

    def sample(self, n_per_class, device="cpu", tau=0.5):
        """Generate n_per_class images per digit using uniform-prior latents.

        z ~ Categorical(1/K): pick one of the n_categories classes uniformly.
        """
        self.eval()
        grid_rows = []
        for label in range(self.n_classes):
            c = F.one_hot(torch.tensor(label), self.n_classes).float().unsqueeze(0)
            c = c.repeat(n_per_class, 1).to(device)
            idx = torch.randint(0, self.n_categories, (n_per_class,))
            z = F.one_hot(idx, self.n_categories).float().to(device)
            with torch.no_grad():
                imgs = self.decoder(torch.cat([z, c], dim=1))
            grid_rows.append(imgs)
        return torch.cat(grid_rows, dim=0)


def categorical_cvae_loss(x_recon, x, logits):
    """Discrete-latent CVAE loss (ELBO negative), reduction='sum'.

    recon = -E[log p(x|z)] via BCE        (MNIST pixels in [0, 1])
    kl    = KL(q(z|x) || Uniform(1/K))     computed from the softmax probs
    """
    recon = F.binary_cross_entropy(x_recon, x, reduction="sum")

    probs = F.softmax(logits, dim=-1)  # q(z|x) over K classes
    k = probs.size(-1)
    kl = torch.sum(probs * torch.log(probs * k))  # sum_i pi_i * log(K * pi_i)

    return recon + kl, recon, kl


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs, batch_size, n_categories = 20, 128, 32
    tau_start, tau_end = 1.0, 0.5  # anneal temperature over training
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

    model = CategoricalCVAE(n_categories=n_categories).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        tau = tau_start * (tau_end / tau_start) ** ((epoch - 1) / (epochs - 1))
        total_recon, total_kl = 0.0, 0.0
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            c = F.one_hot(y, model.n_classes).float().to(device)
            x_recon, logits, _ = model(x, c, tau=tau)
            loss, recon, kl = categorical_cvae_loss(x_recon, x, logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_recon += recon.item()
            total_kl += kl.item()

        n = len(train_loader.dataset)
        print(
            f"epoch {epoch:02d} | tau={tau:.3f} | recon={total_recon / n:.3f} | "
            f"kl={total_kl / n:.3f} | total={total_recon + total_kl:.2f}"
        )

        samples = model.sample(n_per_class=8, device=device)
        save_image(
            samples,
            f"{sample_dir}/discrete_cvae_samples_epoch{epoch:02d}.png",
            nrow=8,
            padding=2,
        )


if __name__ == "__main__":
    main()
```
- Random Choice
```
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class SoftmaxRandomChoiceCVAE(nn.Module):
    """Conditional VAE with a discrete (categorical) latent sampled via the
    "Softmax + Random Choice" scheme.

    Unlike Gumbel-Softmax (which injects Gumbel noise into logits then applies
    softmax), this method:
      1. softmax(logits)      -> categorical probabilities q(z|x)
      2. Random Choice        -> torch.multinomial(probs) gives a hard one-hot z
      3. Straight-Through (ST)-> forward uses the hard one-hot, backward
                                 substitutes the softmax probabilities as the
                                 surrogate gradient (z = z_hard + probs - probs.detach())

    Condition c is the digit label (one-hot, 10-dim), concatenated into both
    the encoder's first FC layer and the decoder's input.
    """

    def __init__(self, n_categories=32, n_classes=10, in_channels=1):
        super().__init__()
        self.n_categories = n_categories
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
        self.fc_logits = nn.Linear(256 + n_classes, n_categories)

        self.decoder = nn.Sequential(
            nn.Linear(n_categories + n_classes, 256),
            nn.ReLU(True),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, logits):
        """Softmax + Random Choice with a straight-through estimator."""
        probs = F.softmax(logits, dim=-1)               # categorical probs
        idx = torch.multinomial(probs, 1).squeeze(-1)   # random choice (non-diff)
        z_hard = F.one_hot(idx, self.n_categories).float()
        # straight-through: forward = z_hard, backward gradient = d(softmax)
        return z_hard + (probs - probs.detach())

    def forward(self, x, c):
                                                                   
           
                                              
        h = self.encoder(x)
        h = torch.cat([h, c], dim=1)
        logits = self.fc_logits(h)
        z = self.reparameterize(logits)
                                                                         
                                                           
                                                                
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc), logits, z

    def sample(self, n_per_class, device="cpu"):
        """Generate n_per_class images per digit using uniform-prior latents.

        z ~ Categorical(1/K): pick one of the n_categories classes uniformly.
        """
        self.eval()
        grid_rows = []
        for label in range(self.n_classes):
            c = F.one_hot(torch.tensor(label), self.n_classes).float().unsqueeze(0)
            c = c.repeat(n_per_class, 1).to(device)
            idx = torch.randint(0, self.n_categories, (n_per_class,))
            z = F.one_hot(idx, self.n_categories).float().to(device)
            with torch.no_grad():
                imgs = self.decoder(torch.cat([z, c], dim=1))
            grid_rows.append(imgs)
        return torch.cat(grid_rows, dim=0)


def categorical_cvae_loss(x_recon, x, logits):
    """Discrete-latent CVAE loss (ELBO negative), reduction='sum'.

    recon = -E[log p(x|z)] via BCE        (MNIST pixels in [0, 1])
    kl    = KL(q(z|x) || Uniform(1/K))     computed from the softmax probs
    """
    recon = F.binary_cross_entropy(x_recon, x, reduction="sum")

    probs = F.softmax(logits, dim=-1)  # q(z|x) over K classes
    k = probs.size(-1)
    kl = torch.sum(probs * torch.log(probs * k))  # sum_i pi_i * log(K * pi_i)

    return recon + kl, recon, kl


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs, batch_size, n_categories = 20, 128, 32
                                                                     
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

    model = SoftmaxRandomChoiceCVAE(n_categories=n_categories).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
                                                                               
        total_recon, total_kl = 0.0, 0.0
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            c = F.one_hot(y, model.n_classes).float().to(device)
            x_recon, logits, _ = model(x, c)
            loss, recon, kl = categorical_cvae_loss(x_recon, x, logits)
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
            f"{sample_dir}/softmax_choice_samples_epoch{epoch:02d}.png",
            nrow=8,
            padding=2,
        )


if __name__ == "__main__":
    main()
```
## 12.GAN
## 13.PositionalEmbedding
## 14.DiffusionModel 
- https://zhuanlan.zhihu.com/p/525106459
- https://www.zhihu.com/question/536012286/answer/2533146567
- DDPM 高质量生成依赖1000步 
![DDPM训练测试算法流程图](https://pic2.zhimg.com/v2-6a41afbb1bf22710efc37646b69ea085_1440w.jpg)
- DDIM
## 15.NormalizedFlow 
- https://medium.com/ai-blog-tw/%E6%B7%B1%E5%85%A5%E6%B7%BA%E5%87%BA-normalizing-flow-nice-realnvp-glow-flow-generative-model%E4%B8%8D%E5%8F%AA%E6%9C%89-gan%E8%B7%9F-vae-29f8e471121
## 16.Transformer
- https://zhuanlan.zhihu.com/p/525106459
## 17.Flow Matching
## 18.DiT
## 19.ViT




