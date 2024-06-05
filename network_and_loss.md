# 目录
- network
- loss
## network

## loss


- entropy of a probability distribution P
```
用来量化从概率分布中得到的事件的平均信息量
Entropy, in the context of a probability distribution, is a measure of the uncertainty or randomness in the distribution. It quantifies the average amount of information needed to represent events drawn from the distribution

熵越大，事件的不确定性越大，或者叫随机性越大。
熵越小，事件的不确定性越小，或者叫随机性越小。

定义： H(P) = -\sum_{x} P(x) \log P(x)   where (x) represents individual events in the distribution.
```


- cross entropy(交叉熵) loss 
```
Cross-entropy loss, also known as log loss, is a loss function used in machine learning and deep learning to measure the difference between two probability distributions.
It is commonly used in classification problems to quantify the difference between predicted probabilities and actual class labels.

// Binary Cross-Entropy Loss:
loss(Y, p) = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]

// Categorical Cross-Entropy Loss
loss(Y, p) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} y_{ij} \log(p_{ij})
```


- KL divergence from Q to P
```
非距离度量，非对称
D_KL(P||Q) = \sum_{x} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
```

- 交叉熵、熵、KL散度之间的关系
```
CrossH(P, Q) = H(P) + D_{KL}(P||Q) 
```  
