# Machine Learning
# 深度学习
- LSTM https://zhuanlan.zhihu.com/p/32085405
- GRU
- RNN
- VAE
   - github Keras_Demo/09xxxx
- Attention Mechanism注意力机制 [知乎](https://zhuanlan.zhihu.com/p/46313756)
   - 权重计算函数
      - 方法1 多层感知机. 主要是先将query和key进行拼接，然后接一个激活函数为tanh的全连接层，然后再与一个网络定义的权重矩阵做乘积。
      - 方法2 Bilnear. 通过一个权重矩阵直接建立q和k的关系映射，比较直接，且计算速度较快。
      - 方法3 Dot Product. 不需要参数，条件是q和k需要维度相同
      - 方法4 Scaled-dot Product. 
   - self-attention
      - 
# 传统机器学习
- Decision Tree
- Random Forest
- Bayes Classifier
- SVM
- 逻辑回归Logistic Regression
- 判别分析
   - 比如LDA
   - 降维的一种， 假设数据服从正太分布
- 提升算法Boosting
   - 比如AdaBoost
   - 例子 opencv
      - [文档](https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html)
      - 基于[2001]Rapid Object Detection using a Boosted Cascade of Simple Features
      - 只处理face
      - 
- 装袋算法Bagging
- 多专家模型
   - 合并多神经网络的结果
- 最大熵模型
- EM
   - [例子](https://www.tuicool.com/articles/Av6NVzy)
- 隐马尔可夫模型HMM
- 条件随机场CRF
- 十大算法
   - [文字表述](https://blog.csdn.net/u011067360/article/details/24368085)
   
# 损失函数
- hinge loss
```
def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)
```
- Multiclass SVM loss
```
def svm_loss(x, y, W):
   score = W.dot(x)
   margins = np.maximum(0, scores - scores[y] + 1)
   margins[y] = 0
   loss_i = np.sum(margins)
   return loss_i
```

