# Machine Learning

# 数据集
- [Microsoft COCO](https://zhuanlan.zhihu.com/p/32566503)
- [Places](http://places2.csail.mit.edu/explore.html)

# 深度学习
- Receptive Field 感受野 及可视化 https://zhuanlan.zhihu.com/p/24833574?utm_source=tuicool&utm_medium=referral

- SPP https://github.com/yhenon/keras-spp
   - SPP
   - RoiPooling ![RoiPooling](https://github.com/liangjin2007/data_liangjin/blob/master/RoiPooling.jpg?raw=true)
   - RoiPoolingConv
- LSTM https://zhuanlan.zhihu.com/p/32085405
- GRU https://zhuanlan.zhihu.com/p/32481747
- RNN https://zhuanlan.zhihu.com/p/32085405
- VAE
   - github Keras_Demo/09xxxx
- Attention Mechanism注意力机制 [知乎](https://zhuanlan.zhihu.com/p/46313756)
   - 权重计算函数
      - 方法1 多层感知机. 主要是先将query和key进行拼接，然后接一个激活函数为tanh的全连接层，然后再与一个网络定义的权重矩阵做乘积。
      - 方法2 Bilnear. 通过一个权重矩阵直接建立q和k的关系映射，比较直接，且计算速度较快。
      - 方法3 Dot Product. 不需要参数，条件是q和k需要维度相同
      - 方法4 Scaled-dot Product. 
   - self-attention
   - keras-attention-mechanism https://github.com/philipperemy/keras-attention-mechanism
      - attention_dense
      - 
- Mixup数据增强机制 
   - 实现简单
   - 具体实现看这里 https://github.com/yu4u/mixup-generator/blob/master/mixup_generator.py
   - 第一步、取得打乱的样本索引；逐对batch进行计算
-弱监督公式
   - 不完全训练
      - 主动学习
      - 半监督学习
   
# 传统机器学习
- 线性回归 Linear Regression
- 逻辑回归 Logistic Regression
- k最近邻 k-Nearest Neighbor
- 支持向量机 SVM
- 基于树的方法
   - 决策树 Decision Tree: 分类， 使用时跟普通的模型一样只要调用tree.fit(x, y)即可。
      - 优点：
      能处理属性是数值的数据或者属性是类目的数据
      预测效率log(N_train)
      能处理多输出问题
      - 缺点：
      过拟合，需要通过pruning
      不稳定
      NP-hard:实际都是贪婪算法或者启发式算法
      数据不平衡则树不平衡
- 判别分析
   - 比如LDA
   - 降维的一种， 假设数据服从正太分布
- 提升算法Boosting
   - 比如AdaBoost
   - 例子 opencv
      - [文档](https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html)
      - 基于[2001]Rapid Object Detection using a Boosted Cascade of Simple Features
      - 只处理face
- 集成方法
   - 回归树 Regression Tree（CART）和集成Ensemble
      -叶子节点有权重的二叉回归树
- 随机森林 Random Forest
   - XGBoost(extreme gradient boost)
      - paper XGBoost: A Scalable Tree Boosting System， 2016； 树集成模型
      - 博客 https://snaildove.github.io/2018/10/02/get-started-XGBoost/
   - GBDT
- 装袋算法Bagging
- 多专家模型
   - 合并多神经网络的结果
- Bayes Classifier
- 最大熵模型
   - 信息量=log(1/概率)=-log(概率)
   意外越大，越不可能发生，概率就越小，信息量也就越大，也就是信息越多。比如说“今天肯定会天黑”，实现概率100%，说了和没说差不多，信息量就是0。理解一下掷骰子比投硬币信息量更大。
   - 熵 = -sum(概率log(概率)),是信息量的期望值， 描述的是意外程度，
   描述的也是意外程度，即不确定性。0<H<log2(m)，m是分类个数，log2(m)是均匀分布时的熵。二分类熵的取值范围是[0,1]，0是非常确定，1是非常不确定。
   - 信息量与熵
   分类越多->信息量越大->熵越大，如图所示：图Ｃ将点平均分成5类（熵为2.32），图B将点平均分成两类（熵为1），则看起来Ｃ更复杂，更不容易被分类，熵也更大。
   分类越平均->熵越大。图Ｂ（熵为1）比Ａ（熵为0.72）更复杂，更不容易被分类，熵也更大
   - 信息增益
   信息增益(Information Gain)：熵A-条件熵B，是信息量的差值。也就是说，一开始是Ａ，用了条件后变成了Ｂ，则条件引起的变化是A-B，即信息增益（它描述的是变化Delta）。好的条件就是信息增益越大越好，即变化完后熵越小越好（熵代表混乱程度，最大程度地减小了混乱）。因此我们在树分叉的时候，应优先使用信息增益最大的属性，这样降低了复杂度，也简化了后边的逻辑。
   - 举例
   假设使用8天股票数据实例，以次日涨/跌作为目标分类，红为涨，蓝为跌，如上图所示涨跌概率各50%:50%（2分类整体熵为1），有D,E,F三个属性描述当日状态，它们分别将样本分为两类：方和圆，每类四个。D中方和圆中涨跌比例各自为50%:50%（条件熵为1，信息增益0）。E中方的涨跌比例为25%:75%，圆的涨跌比例为75%:25%（条件熵为0.81，信息增益0.19），F中方的涨跌比例为0:%:100%，圆的涨跌比例为100%:0%（条件熵为0，信息增益1）。
    我们想要寻找的属性是可直接将样本分成正例和反例的属性，像属性F为圆一旦出现，第二天必大涨，而最没用的是D，分类后与原始集合正反比例相同。E虽然不能完全确定，也使我们知道当E为圆出现后，比较可能涨，它也带有一定的信息。
    使用奥卡姆剃刀原则：如无必要，勿增实体。不确定有用的就先不加，以建立最小的树。比如，如个属性X（代表当日涨幅），明显影响第二天，则优先加入，属性Y（代表当天的成交量），单独考虑Y，可能无法预测第二天的涨跌，但如果考虑当日涨幅X等因素之后，成交量Y就可能变为一个重要的条件，则后加Y。属性Z（隔壁张三是否买了股票），单独考虑Z，无法预测，考虑所有因素之后，Z仍然没什么作用。因此属性Z最终被丢弃。策略就是先把有用的挑出来，不知道是不是有用的往后放。
   - 熵的作用
   熵是个很重要的属性，它不只是在决策树里用到，各个分类器都会用到这个量度。比如说，正例和反例为99:1时，全选正例的正确率也有99%，这并不能说明算法优秀。就像在牛市里能挣钱并不能说明水平高。另外分成两类，随机选的正确率是50%；分而三类，则为33%，并不是算法效果变差了。在看一个算法的准确率时，这些因类都要考虑在内。在多个算法做组合时，也应选择信息增益大的放在前面。
   在决策树中利用熵，可以有效地减小树的深度。计算每种分类的熵，然后优先熵小的，依层次划分数据集。熵的算法，一般作为决策树的一部分，把它单拿出来，也可以用它筛选哪个属性是最直接影响分类结果的。

- EM
   - [例子](https://www.tuicool.com/articles/Av6NVzy)
- 隐马尔可夫模型HMM
- 条件随机场CRF
- 十大算法
   - [文字表述](https://blog.csdn.net/u011067360/article/details/24368085)

- Active Learning


# 损失函数
- hinge loss即svm loss
中文翻译为合页损失函数,具体在知乎上有解释https://www.zhihu.com/question/47746939?sort=created
普通样本对应于hingeloss的零区域,所以所有的普通样本都不参与最终超平面的决定。
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
- softmax loss
```
def softmax_loss(y_true, y_pred):
   return -log(y_pred[one_hot_to_category(y_true)])
```
- KL Divergence loss
```
def KLDivergence_loss(y_true, y_pred):
   y_true = K.clip(y_true, epsilon(), 1.0-epsilon())
   res = K.sum(y_true*(K.log(y_true)-K.log(y_pred)) # note that it equals to simper -log 
   return res
```
   
- cross entropy loss
```
def cross_entropy_loss(y_true, y_pred):
   return -K.sum(y_true*log(y_pred)) # actually = -log(pi)
```

- lasso L1 Loss

- L2 Loss

