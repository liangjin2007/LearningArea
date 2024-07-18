- https://zhuanlan.zhihu.com/p/348593638
# 目录
- Transformer
- Vision Transformer
- 

## Transformer
- sequence to sequence model
- 背景：解决RNN无法并行预测的问题。
```
SelfAttentionLayer
  q, k, v, alpha, a^i, b^i=sum alpha_ij vj, alpha_ij = softmax(qi.dot(kij)/sqrt(dim)), Add&Norm layer, Residual block.
  Multi-head attention layer
  FFN
Encoder
Decoder
问题： 为啥Decoder仍然是时序依赖的？LLM是怎么训练的？
```
- 代码实现解读
