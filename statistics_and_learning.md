# 统计，学习相关

- [代码](https://github.com/Dod-o/Statistical-Learning-Method_Code)
 
#### 前言
对特征的理解: 如果输入点是三维点(x, y, z) 表示样本点有三个特征。

#### 第一章、统计学习方法概论
这里有一堆概念，从概率论，统计学习，监督学习，模型，策略，算法，模型评估，模型选择，误差，过拟合，欠拟合，正则化，交叉验证，泛化能力，生成模型与判别模型，分类问题，标注问题，回归问题等进行说明。

#### 第二章、感知机
二分类的线性分类模型。 [perceptron_dichotomy.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/perceptron/perceptron_dichotomy.py)

#### 第三章、k邻近法
10分类的例子，没有训练过程。对每个测试集的样本从训练集找k个最近的样本，以投票唱票的方式获取label。[KNN.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/KNN/KNN.py)
程序中没有实现kdtree加速最近的k个点的搜索。


#### 第四章、朴素贝叶斯法
   - 基于贝叶斯定理和条件独立的假设学习(X, Y)的联合分布，然后对于输入的x, 根据贝叶斯定理求出后验概率最大的y。朴素贝叶斯法实现简单，学习与预测的效率都很高。
   - 模型
      - 把X, Y 都看成随机变量。
      - n个样本看成n次投币。
      - 通过训练数据学习联合概率。具体体先学习P(Y=ck)，然后学习P(X=x|Y=ck)=P(x(1) = x(1), ...|Y=ck)。由条件独立特性可简化为概率连乘。
      - 可看成是一个生成模型。
      - 预测是给定x，然后通过生成模型求后验概率P(Y=ck|X=x), 把后验概率最大的作为类别输出。
      - 后验概率最大化等价于期望风险最小化。
      - 1.先验概率： P(Y=ck) = sum(I(yi=ck), i in [1, N])/N
      - 2.条件概率： P(X|Y=ck)
   - 策略：
      - 极大似然估计 for 1 and 2
      - 贝叶斯估计 for 1 and 2
      - max后验概率
   - 算法：
      - 显示计算
   - 跟knn类似，没有实际训练过程。需要用矩阵把先验概率和条件概率存好。然后用后验概率计算出最终的概率。
   
   
- 第五章、决策树 这里讲分类决策树
   - 决策树学习通常包含三个步骤：特征选择、决策树生成和决策树的修剪。
   - ID3, C4.5, CART算法
   - 树形结构，node, directed edge, internal node,  leaf node
   - 决策树可看成一个if-then规则的集合。在每一条路径上构建一条规则。
   - 内部节点表示规则的条件，叶子节点表示规则的结论
   - 特征空间划分及区域条件概率分布。
   - 决策树对应于条件概率分布。
   - 特征选择
      - 用特征进行分类比随机分类效果要好，反之有些特征没什么用的话就称为没什么分类能力，经验上需要扔掉这样的特征。
      - 决定哪个特征来划分特征空间
      
- 第六章、逻辑回归与最大熵
   - 最大熵原理： 在所有可能的概率模型中，熵最大的模型是最好的模型。H(P) = -Sum(P(xi)logP(xi))。满足0 <= H(P) <= log|X|。 |X|指X的取值个数。
   - 极大似然估计: max Mult P(xi)
- 第七章、支持向量机
- 第八章、提升方法
- 第九章、EM算法及其推广
- 第十章、隐马尔可夫模型
- 第十一章、条件随机场
- 第十二章、总结



# Machine Learning
http://www.vlfeat.org/overview/
http://www.cse.psu.edu/~rtc12/CSE586/lectures/meanshiftclustering.pdf
- mean-shift
   - Objective : Find the densest region

# 如何训练
- 欠拟合和过拟合：欠拟合是指模型不能在训练集上获得足够低的误差。而过拟合是指训练误差和和测试误差之间的差距太大，误差即loss。
- 误差的95%置信区间： （误差均值-1.96误差标准差，误差均值+1.96误差标准差)
- 

# 网络架构
- resnet50 keras自带
- densenet https://github.com/flyyufelix/DenseNet-Keras
- inceptionv3

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

- Mixup数据增强机制 
   - 实现简单
   - 具体实现看这里 https://github.com/yu4u/mixup-generator/blob/master/mixup_generator.py
   - 第一步、取得打乱的样本索引；逐对batch进行计算

- 弱监督学习
   - 主动学习：假定没打标的实例的GT标签能通过专家查询得到，目标之一是极小化查询次数。给定一个比较小的打标好的数据集，主动学习会从没打标的数据中选择最有价值的实例进行专家查询。
      - 信息量：减少统计模型的uncertainty
      - 代表性：尽量好地表示输入模式的结构
   - 半监督学习
      - 数据分布假定
         - 聚类假定
         - 流形假定
      - 方法
         - 生成方法:
            假定
         - 基于图的方法
         - low-density separation method
         - disagreement-based method
      - 
- GAN
   - CVPR2018 Tutorials on GANs https://sites.google.com/view/cvpr2018tutorialongans/
- Model Ensemble
   - [Model Ensemble](http://cs231n.github.io/neural-networks-3/#ensemble)
- EMA
   - https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
- SMA
   - 
   - 
   
- 对抗样本问题
```
[2014]EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES 提出了流行的FGSM方法
[2017]Feature Squeezing:Detecting Adversarial Examples in Deep Neural Networks
```
对抗样本论文汇总 https://zhuanlan.zhihu.com/p/32037178

- 对抗训练

- CRF
mean field approximation http://vladlen.info/papers/densecrf-supplementary.pdf

# 传统机器学习
- 线性回归 Linear Regression
- 逻辑回归 Logistic Regression
- k最近邻 k-Nearest Neighbor
- 支持向量机 SVM

- 基于树的方法
   - Obj(Θ)=L(Θ)+Ω(Θ), 在上式中 L(Θ) 代表的是训练误差，表示该模型对于训练集的匹配程度。Ω(Θ) 代表的是正则项，表明的是模型的复杂度
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
   - 决策回归树CART和集成模型
      - 决策规则与决策树的一样。
      - 每个叶子节点上都包含了一个权重，也有人叫做分数。
      - 回归树的集成模型Ensemble sum(fk(xi))
      - 模型和参数：学习的是每棵树的结构和每片叶子的权重（分数）
      ![parameters](https://github.com/liangjin2007/data_liangjin/blob/master/decisiontree_models.jpg?raw=true)
      - 在单一变量上学习一棵树：定义一个目标函数，优化它 
      ![step function example](https://github.com/liangjin2007/data_liangjin/blob/master/decisiontree_stepfunction.jpg?raw=true)
      - 目标函数 vs 启发式
         - 启发式: 按信息增益；对树剪枝；最大深度；对叶子节点进行平滑
         - 启发式映射到目标函数：
            - 信息增益->训练误差
            - 剪枝->按照树节点对数目定义对正则化项
            - 最大深度 -> 限制函数空间
            - 对叶子值进行平滑操作->叶子权重对L2正则化项
      - 回归树不仅仅用于回归
         - 还可以用于分类，排序，取决于怎么定义目标函数
   - 梯度提升Gradient Boosting
   - GBDT
      - 在函数空间中利用梯度下降法进行优化, http://wepon.me/files/gbdt.pdf
      - XGBoost
         - 在函数空间中用牛顿法进行优化

- 判别分析
   - 比如LDA
   - 降维的一种， 假设数据服从正太分布
   
- 集成方法Ensemble Methods <Foundations and Algorithms>
   - Boosting： 一系列弱学习器组合成强学习器

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

- Focal Loss

- Triplet Loss

- Contrast Loss
```
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

```


# Keras

## Keras实现多输入多输出

- 模型定义的输入输出需要等量：model definition should contain equal amount of inputs and outputs.
    - 可以添加空输入，空输出，loss使用记录（self.）的方式。

- 数据生成器格式：data generator must output [input1, input2, input3, input4], [output1, output2, output3, output4]而不是[input1,output1],[input2,output2],...

- 输出名：outputs'name can be used to determine which output metrics applying to.

- 写loss函数可以将参数设成y_true, y_pred都是one_hot，只要让多输出的输出设为定义的空的输出，比如：
```
        input_image = Input(shape=input_shape, dtype=tf.float32) # image
        
        input_target_tripletloss = Input(shape=(num_classes,), dtype=tf.float32) # an input for the 2nd loss
        output_target_tripletloss= Lambda(lambda x:x, name="tripletloss")(input_target_tripletloss)

        input_target_centerloss = Input(shape=(1,), dtype=tf.int32) # an input for the 3rd loss
        output_target_centerloss = Lambda(lambda x:tf.squeeze(tf.one_hot(tf.cast(x, tf.int32), num_classes, 1.0, 0.0, dtype=tf.float32),axis=1), name="centerloss")(input_target_centerloss)

        input_target_consistencyloss = Input(shape=(num_classes,), dtype=tf.float32) # an input for the 3rd loss
        output_target_consistencyloss = Lambda(lambda x:x, name="consistencyloss")(input_target_consistencyloss)

        self.centers = Embedding(num_classes, output_dim=fc_size)(input_target_centerloss)

        input_noise_student = GaussianNoise(FLAGS.input_noise)(input_image)
        input_noise_teacher = GaussianNoise(FLAGS.input_noise)(input_image)
       
        resnet_out_student = ResNet50(input_tensor = input_noise_student, include_top=False, input_shape=input_shape, weights=None, tag="student")
        resnet_out_teacher = ResNet50(input_tensor = input_noise_teacher, include_top=False, input_shape=input_shape, weights=None, tag="teacher")
       
        x = GlobalMaxPooling2D()(resnet_out_student.output)
        embedding_student = Dense(fc_size, activation='relu', name = "embedding_student")(x)
        x = Dropout(FLAGS.dropout)(embedding_student)   
        self.probs_student = Dense(num_classes, activation='softmax', name="probs_student")(x)
        self.normed_embedding_student =  Lambda(lambda x: K.l2_normalize(x, axis=1))(embedding_student)
         
        self.student = Model(inputs=input_image, outputs=self.probs_student)
    
        x = GlobalAveragePooling2D()(resnet_out_teacher.output)
        embedding_teacher = Dense(fc_size, activation='relu')(x)
        x = Dropout(FLAGS.dropout)(embedding_teacher)
        self.probs_teacher = Dense(num_classes, activation='softmax', name="probs_teacher")(x)
        self.teacher = Model(inputs=input_image, outputs=self.probs_teacher)
        
        if use_gpu:
            self.mean_teacher = multi_gpu_model(Model(
                   inputs=[input_image, input_target_tripletloss, input_target_centerloss, input_target_consistencyloss]
                , outputs=[self.probs_student, output_target_tripletloss, output_target_centerloss, output_target_consistencyloss]
                )
            ,self.gpu_count)
        else:
            self.mean_teacher = Model(
                inputs=[input_image, input_target_tripletloss, input_target_centerloss, input_target_consistencyloss]
                , outputs=[self.probs_student, output_target_tripletloss, output_target_centerloss, output_target_consistencyloss]
                )
```

```
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import numpy as np
import keras
main_input = Input((100,), dtype='int32', name='main_input')

x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)
aux_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
aux_input = Input((5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, aux_input])
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])
model.compile(optimizer='rmsprop',
loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
loss_weights={'main_output': 1., 'aux_output': 0.3})
print(model.summary())
```

## Keras实现复杂模型

https://www.imooc.com/article/details/id/31034
```
from keras.layers import Input,Embedding,LSTM,Dense,Lambda
from keras.layers.merge import dot
from keras.models import Model
from keras import backend as K

word_size = 128
nb_features = 10000
nb_classes = 10
encode_size = 64
margin = 0.1

embedding = Embedding(nb_features,word_size)
lstm_encoder = LSTM(encode_size)

def encode(input):
    return lstm_encoder(embedding(input))

q_input = Input(shape=(None,))
a_right = Input(shape=(None,))
a_wrong = Input(shape=(None,))
q_encoded = encode(q_input)
a_right_encoded = encode(a_right)
a_wrong_encoded = encode(a_wrong)

q_encoded = Dense(encode_size)(q_encoded) #一般的做法是，直接讲问题和答案用同样的方法encode成向量后直接匹配，但我认为这是不合理的，我认为至少经过某个变换。

right_cos = dot([q_encoded,a_right_encoded], -1, normalize=True)
wrong_cos = dot([q_encoded,a_wrong_encoded], -1, normalize=True)

loss = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([wrong_cos,right_cos])

model_train = Model(inputs=[q_input,a_right,a_wrong], outputs=loss)
model_q_encoder = Model(inputs=q_input, outputs=q_encoded)
model_a_encoder = Model(inputs=a_right, outputs=a_right_encoded)

model_train.compile(optimizer='adam', loss=lambda y_true,y_pred: y_pred)
model_q_encoder.compile(optimizer='adam', loss='mse')
model_a_encoder.compile(optimizer='adam', loss='mse')

model_train.fit([q,a1,a2], y, epochs=10)
#其中q,a1,a2分别是问题、正确答案、错误答案的batch，y是任意形状为(len(q),1)的矩阵
```

```
from keras.layers import Input,Conv2D, MaxPooling2D,Flatten,Dense,Embedding,Lambda
from keras.models import Model
from keras import backend as K

nb_classes = 100
feature_size = 32

input_image = Input(shape=(224,224,3))
cnn = Conv2D(10, (2,2))(input_image)
cnn = MaxPooling2D((2,2))(cnn)
cnn = Flatten()(cnn)
feature = Dense(feature_size, activation='relu')(cnn)
predict = Dense(nb_classes, activation='softmax', name='softmax')(feature) #至此，得到一个常规的softmax分类模型

input_target = Input(shape=(1,))
centers = Embedding(nb_classes, feature_size)(input_target) #Embedding层用来存放中心
l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]), 1, keepdims=True), name='l2_loss')([feature,centers])

model_train = Model(inputs=[input_image,input_target], outputs=[predict,l2_loss])
model_train.compile(optimizer='adam', loss=['sparse_categorical_crossentropy',lambda y_true,y_pred: y_pred], loss_weights=[1.,0.2], metrics={'softmax':'accuracy'})

model_predict = Model(inputs=input_image, outputs=predict)
model_predict.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_train.fit([train_images,train_targets], [train_targets,random_y], epochs=10)
#TIPS：这里用的是sparse交叉熵，这样我们直接输入整数的类别编号作为目标，而不用转成one hot形式。所以Embedding层的输入，跟softmax的目标，都是train_targets，都是类别编号，而random_y是任意形状为(len(train_images),1)的矩阵。
```


## Keras pretrained_word_embeddings.py

- 文本处理之词向量
    - word2vec
    - GloVe
- Keras如何表示词向量
    - embedding_matrix
    - one_hot独日编码

- https://blog.csdn.net/sinat_22510827/article/details/90727435

## Keras mnist_acgan.py
- 可以做分类的GAN
- 自定义训练循环





## Keras mnist_siamese.py
两个点：1.loss数与ouput数一致 2. 如果有两个输入

```
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)
```


## Keras mnist_swwae.py
s means stacked 堆叠
MaxPooling, UpSampling2D
需要求梯度的梯度，backend需要是theano。
对这两个Layer的参数比较难理解


## ImageDataGenerator的用法
datagen.flow(x_train, y_train, batch_size=batch_size)


## Keras Antirectifier 如何自定义Layer
We need to specify two methods: `compute_output_shape` and `call`.
```
class Antirectifier(layers.Layer):
    '''This is the combination of a sample-wise
    L2 normalization with the concatenation of the
    positive part of the input with the negative part
    of the input. The result is a tensor of samples that are
    twice as large as the input samples.

    It can be used in place of a ReLU.

    # Input shape
        2D tensor of shape (samples, n)

    # Output shape
        2D tensor of shape (samples, 2*n)

    # Theoretical justification
        When applying ReLU, assuming that the distribution
        of the previous output is approximately centered around 0.,
        you are discarding half of your input. This is inefficient.

        Antirectifier allows to return all-positive outputs like ReLU,
        without discarding any data.

        Tests on MNIST show that Antirectifier allows to train networks
        with twice less parameters yet with comparable
        classification accuracy as an equivalent ReLU-based network.
    '''

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] *= 2
        return tuple(shape)

    def call(self, inputs):
        inputs -= K.mean(inputs, axis=1, keepdims=True)
        inputs = K.l2_normalize(inputs, axis=1)
        pos = K.relu(inputs)
        neg = K.relu(-inputs)
        return K.concatenate([pos, neg], axis=1)
```

## Keras mnist net2net用知识迁移来加速训练
```
算是迁移学习的一种，用更宽的卷积去套住已经训练好的权重
```


## Keras mnist denoising autodecoder
```
'''Trains a denoising autoencoder on MNIST dataset.

Denoising is one of the classic applications of autoencoders.
The denoising process removes unwanted noise that corrupted the
true signal.

Noise + Data ---> Denoising Autoencoder ---> Data

Given a training dataset of corrupted data as input and
true signal as output, a denoising autoencoder can recover the
hidden structure to generate clean data.

This example has modular design. The encoder, decoder and autoencoder
are 3 models that share weights. For example, after training the
autoencoder, the encoder can be used to  generate latent vectors
of input data for low-dim visualization like PCA or TSNE.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(1337)

# MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Generate corrupted MNIST images by adding noise with normal dist
# centered at 0.5 and std=0.5
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
latent_dim = 16
# Encoder/Decoder number of CNN layers and filters per layer
layer_filters = [32, 64]

# Build the Autoencoder Model
# First build the Encoder Model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# Stack of Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use MaxPooling2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

# Shape info needed to build Decoder Model
shape = K.int_shape(x)

# Generate the latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# Instantiate Encoder Model
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# Build the Decoder Model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# Stack of Transposed Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use UpSampling2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

x = Conv2DTranspose(filters=1,
                    kernel_size=kernel_size,
                    padding='same')(x)

outputs = Activation('sigmoid', name='decoder_output')(x)

# Instantiate Decoder Model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder Model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')

# Train the autoencoder
autoencoder.fit(x_train_noisy,
                x_train,
                validation_data=(x_test_noisy, x_test),
                epochs=30,
                batch_size=batch_size)

# Predict the Autoencoder output from corrupted test images
x_decoded = autoencoder.predict(x_test_noisy)

# Display the 1st 8 corrupted and denoised images
rows, cols = 10, 30
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()

```

## Keras transfer learning
```
layer.trainable = False
```

## Keras mnist mlp multiple layer perception多层感知器

## Keras tfrecords

## Keras sklearn wrapper
```
dense_size_candidates = [[32], [64], [32, 32], [64, 64]]
my_classifier = KerasClassifier(make_model, batch_size=32)
validator = GridSearchCV(my_classifier,
                         param_grid={'dense_layer_sizes': dense_size_candidates,
                                     # epochs is avail for tuning even when not
                                     # an argument to model building function
                                     'epochs': [3, 6],
                                     'filters': [8],
                                     'kernel_size': [3],
                                     'pool_size': [2]},
                         scoring='neg_log_loss',
                         n_jobs=1)
validator.fit(x_train, y_train)

print('The parameters of the best model are: ')
print(validator.best_params_)

# validator.best_estimator_ returns sklearn-wrapped version of best model.
# validator.best_estimator_.model returns the (unwrapped) keras model
best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(x_test, y_test)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)
```

## Keras cifar10_cnn_tfaugment2d.py

- tf.contrib.image.compose_transforms(*transforms)
- tf.contrib.image.transform()
- tf.contrib.image.angles_to_projective_transforms
- tf.tile
- tf.convert_to_tensor
- tf.expand_dims
- tf.where
- tf.random_uniform
- tf.less

## Keras数据增强的用法

```
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)
```


## Keras SPP(SpatialPyramidPooling)

## Keras caption_generator 代码阅读
https://github.com/anuragmishracse/caption_generator
paper: 
- 安装
    - pip install -r requirements.txt
- 网络架构
    - ![lstm image caption](https://github.com/liangjin2007/data_liangjin/blob/master/model_lstm.png?raw=true)
    - Embedding层，只能用作模型的第一层，输入尺寸为(batch_size, sequence_length)，输出为(batch_size, sequence_length, output_dim)
- 训练数据集
Flickr_8k.trainImages.txt
```
2513260012_03d33305cf.jpg
2903617548_d3e38d7f88.jpg
3338291921_fe7ae0c8f8.jpg
488416045_1c6d903fe0.jpg
2644326817_8f45080b87.jpg
...
```
flickr_8k_train_dataset.txt
```
image_id	captions
2513260012_03d33305cf.jpg	<start> A black dog is running after a white dog in the snow . <end>
2513260012_03d33305cf.jpg	<start> Black dog chasing brown dog through snow <end>
2513260012_03d33305cf.jpg	<start> Two dogs chase each other across the snowy ground . <end>
2513260012_03d33305cf.jpg	<start> Two dogs play together in the snow . <end>
2513260012_03d33305cf.jpg	<start> Two dogs running through a low lying body of water . <end>
2903617548_d3e38d7f88.jpg	<start> A little baby plays croquet . <end>
2903617548_d3e38d7f88.jpg	<start> A little girl plays croquet next to a truck . <end>
2903617548_d3e38d7f88.jpg	<start> The child is playing croquette by the truck . <end>
2903617548_d3e38d7f88.jpg	<start> The kid is in front of a car with a put and a ball . <end>
2903617548_d3e38d7f88.jpg	<start> The little boy is playing with a croquet hammer and ball beside the car . <end>
...
```
Flickr8k.token.txt
```
1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg#1	A girl going into a wooden building .
1000268201_693b08cb0e.jpg#2	A little girl climbing into a wooden playhouse .
1000268201_693b08cb0e.jpg#3	A little girl climbing the stairs to her playhouse .
1000268201_693b08cb0e.jpg#4	A little girl in a pink dress going into a wooden cabin .
1001773457_577c3a7d70.jpg#0	A black dog and a spotted dog are fighting
1001773457_577c3a7d70.jpg#1	A black dog and a tri-colored dog playing with each other on the road .
1001773457_577c3a7d70.jpg#2	A black dog and a white dog with brown spots are staring at each other in the street .
1001773457_577c3a7d70.jpg#3	Two dogs of different breeds looking at each other on the road .
1001773457_577c3a7d70.jpg#4	Two dogs on pavement moving toward each other .
...
```
- 概念
    - 标注 描述图片的一句话
    - 训练集和测试集中每张图片可以有多个标注
    - vocab_size 单词数
    - caps 所有单词
    - max_cap_len  图像标注最多包含几个单词，也就是一句话最多多少个单词（40）。
    - 需要把单词嵌入到一个高维空间中，且保证相近到词在这个空间中比较近。
    - Embedding(vocab_size, 256, input_length=self.max_cap_len), 输入为(None，40)的整数矩阵，整数值为[0,vocab_size-1]。输出为（None，40， 256)
    - 一张图片
    - 每句标注<start>, <end>
    
## Keras R-CNN代码阅读

- 一些Python/Keras相关的使用技巧
取整，类型转化，张量与list相互转化，张量与numpy相互转化， broadcast（非常难），张量reshape（expand_dims, reshape(-1, 1)）；
将二维数组压平

- 继承Model：只需重载__init__, compile, predict三个方法
- 继承Layer: 只需重载__init__, build, call, compute_output_shape, get_config, 
- 字典如何拼接
```
def get_config(self):
    configuration = {
        "maximum_proposals": self.maximum_proposals,
        "minimum_size": self.minimum_size,
        "stride": self.stride
    }
    return {**super(ObjectProposal, self).get_config(), **configuration}
```
- broadcast
broadcast例子：形状为(1,1)的张量减去形状为（3,）的张量结果为(1, 3)的张量。
(1，1)+(3,)=> (1，3) # 左侧是更短的数组，右侧更长，所以先把短的扩充到跟长的一样长。
(1,1,1)+(3,) => (1,1,3)
(1,1,2)+(3,) => broadcast error
(15,4)+(196,1,4) ==> (196,15,4)
```
General Broadcasting Rules¶
When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions, and works its way forward. Two dimensions are compatible when
they are equal, or
one of them is 1
If these conditions are not met, a ValueError: operands could not be broadcast together exception is thrown, indicating that the arrays have incompatible shapes. The size of the resulting array is the maximum size along each dimension of the input arrays.
```

- 基于Faster RCNN实现
- 训练集数据格式
- generator如何写
- 概念
    - boxes: box 1x4 , [lbx, lby, rtx, rty], lbx range in [0, 1]
    - iou threshold: 0.5, i means intersection, u means union
    - anchor: [box1， box2, box3, ...], w, h,x_ctr, y_ctr, size=w*h, size_ratios = size*ratios, ws.K.sqrt(size_ratios)
        - ratio = w/h
        - scales
        - result: (15, 4), 15 boxes each anchor. each row is lx, ly, rx, ry
    - anchor_size: 8 or 16
    - anchor_aspect_ratio
    - anchor_aspect_scale
    
- 网络输入输出及架构
    - 输入
    inputs=[target_bounding_boxes, target_categories, target_image, target_mask, target_metadata]
    - 输出
    outputs=[output_bounding_boxes, output_categories] # batch_size* [maximum_proposals*4, maximum_proposals]
    - VGG
    - 3x3: Conv(64) 输出(None, 14, 14, 64)
    - deltas1: Conv 3x3 (None, 14, 14, 32)
    - scores1: Conv 1x1 (None, 14, 14, 9)
    - target_bounding_boxes: Input (None, None, 4)
    - target_metadata: Input (None, 3), None=>1 # 记录原图尺寸
    - 
    - Anchor层
        - 通过类似于numpy的操作，产生候补boundingboxes
        - 判断inside
        - 根据metadata及padding信息裁剪boundingboxes
        - 根据anchors, indices_inside, targets得到与anchors重叠的gt boxes
            - 用到了类似于numpy的broadcast
        -
    - RPN层
        - 只是添加了两个loss函数
    - ObjectProposal
    - 训练
    - 预测
        - target_bounding_boxes默认值形状: (image_count, 1, 4), 值为0
        - target_categories默认值形状： (image_count, 1, n_categories)， 值为0
        - target_mask默认值形状： (1, 1, 28, 28)，值为0
        - target_metadata默认值形状: 形状为(1, 3)值为[[224, 224, 1.0]]
    - 


## Keras 新闻分类,词向量，Embedding

## Keras Sort
```
K.arange, 
K.expand_dims
K.stack
K.shape
tf.argsort(values)
tf.gather(values, tf.argsort(values))

def spearman_loss(y_true, y_pred):
    n = K.shape(y_pred)[0]
    c = tf.contrib.framework.argsort(y_true[:, 0])
    d = tf.contrib.framework.argsort(y_pred[:, 0])
    return 1.0 - (K.sum(K.square(c-d))*6.0/(K.pow(n, 3)-n))
```

## Keras_Demo笔记
- GAN and VAE
- [](https://spaces.ac.cn/archives/5253/comment-page-1)
- VGG16
    ```
    from keras.applications.vgg16 import preprocess_input,decode_predictions
    model = VGG16(include_top=True, weights='imagenet')
    preds = model.predict(x)
    # 将结果解码成一个元组列表（类、描述、概率）（批次中每个样本的一个这样的列表）
    print('Predicted:', decode_predictions(preds, top=3)[0])
    ```
- utils
    ```
    from keras.utils import np_utils

    np_utils.to_categorical(y_train, nb_classes)

    from keras.utils.vis_utils import plot_model
    ```
- datasets
    ```
    from keras.datasets import mnist
    from keras.datasets import cifar10

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    ```
- network
    ```
    model.add(Flatten()) 可以把多维结构压平成一维
    model.add(Activation('relu')) 激活可以是单独一层
    ```

- Imagenet prediction to class and score
    - Various sorting
    ```
    CLASS_INDEX = None
    CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
    fpath = get_file('imagenet_class_index.json',
                     CLASS_INDEX_PATH,
                     cache_subdir='models',
                     file_hash='c2c37ea517e94d9795004a39431a14cb')
    CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    ```
- numpy方法归一化

    ```
    x -= x.mean()
    x /= (x.std()+1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    ```

- 用Callback画acc_loss曲线
    ```
    #写一个LossHistory类，保存loss和acc
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = {'batch': [], 'epoch': []}
            self.accuracy = {'batch': [], 'epoch': []}
            self.val_loss = {'batch': [], 'epoch': []}
            self.val_acc = {'batch': [], 'epoch': []}

        def on_batch_end(self, batch, logs={}):
            self.losses['batch'].append(logs.get('loss'))
            self.accuracy['batch'].append(logs.get('acc'))
            self.val_loss['batch'].append(logs.get('val_loss'))
            self.val_acc['batch'].append(logs.get('val_acc'))

        def on_epoch_end(self, batch, logs={}):
            self.losses['epoch'].append(logs.get('loss'))
            self.accuracy['epoch'].append(logs.get('acc'))
            self.val_loss['epoch'].append(logs.get('val_loss'))
            self.val_acc['epoch'].append(logs.get('val_acc'))

        def loss_plot(self, loss_type):
            iters = range(len(self.losses[loss_type]))
            #创建一个图
            plt.figure()
            # acc
            plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
            # loss
            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
            if loss_type == 'epoch':
                # val_acc
                plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
                # val_loss
                plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.grid(True)#设置网格形式
            plt.xlabel(loss_type)
            plt.ylabel('acc-loss')#给x，y轴加注释
            plt.legend(loc="upper right")#设置图例显示位置
            plt.show()
    ```

- visualize convolution filter
   - 如何将一个网络中的某卷积层作为loss
   - 使滤波器的激活最大化，输入图像就是要可视化的滤波器 ？ 
   - 构建损失函数，计算输入图像的梯度。损失函数
   - 如何自己写梯度更新
   - 如何对多维数据根据某一维排序
   - 如何把多个小图片堆积到一起显示
    ```
    #说明：通过在输入空间的梯度上升，可视化VGG16的滤波器。
    from scipy.misc import imsave
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from keras.applications import vgg16
    from keras import backend as K
    img_width = 128
    img_height = 128
    layer_name = 'block5_conv1'

    #将张量转换成有效图像
    def deprocess_image(x):
        # 对张量进行规范化
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1
        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)
        # 转化到RGB数组
        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')
        return x
    def normalize(x):
        # 效用函数通过其L2范数标准化张量
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    model = vgg16.VGG16(weights='imagenet', include_top=False)
    print('Model loaded.')
    model.summary()

    input_img = model.input
    #用一个字典layer_dict存放Vgg16模型的每一层
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    print(layer_dict.keys())
    print(layer_dict.values())

    #提取滤波器
    #通过梯度上升，修改输入图像，使得滤波器的激活最大化。这是的输入图像就是我们要可视化的滤波器。
    
    kept_filters = []
    for filter_index in range(0, 200):
        # 我们只扫描前200个滤波器，
        # 但实际上有512个
        print('Processing filter %d' % filter_index)
        start_time = time.time()
        # 我们构建一个损耗函数，使所考虑的层的第n个滤波器的激活最大化
        #由字典索引输出
        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, filter_index])

        # 我们计算输入图像的梯度与这个损失
        grads = K.gradients(loss, input_img)[0]

        # 归一化技巧：我们规范化梯度
        grads = normalize(grads)

        # 此函数返回给定输入图像的损耗和梯度
        # inputs: List of placeholder tensors.
        # outputs: List of output tensors.
        iterate = K.function([input_img], [loss, grads])

        # 梯度上升的步长
        step = 1.

        # 我们从带有一些随机噪声的灰色图像开始
        input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # 我们运行梯度上升20步
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            #每次上升一步，逐次上升
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # 一些过滤器陷入0，我们可以跳过它们
                break

        # 解码所得到的输入图像
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            #将解码后的图像和损耗值加入列表
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))


    # 我们将在8 x 8网格上选择最好的64个滤波器。
    n = 8
    # 具有最高损失的过滤器被假定为更好看。
    # 我们将只保留前64个过滤器。
    #Lambda:本函数用以对上一层的输出施以任何Theano/TensorFlow表达式
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # 构建一张黑色的图片，有足够的空间
    # 我们的尺寸为128 x 128的8 x 8过滤器，中间有5px的边距
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # 使用我们保存的过滤器填充图片
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    #显示滤波器
    plt.imshow(img)
    plt.show()
    plt.imshow(stitched_filters)
    plt.show()
    # 保存结果,将数组保存为图像
    imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
    ```
- LSTM使用，长短期记忆，一种特殊的循环神经网络RNN
    - 将图像高度作为time
    - 知乎上的图解 https://zhuanlan.zhihu.com/p/32085405
    ```
    nb_lstm_outputs = 30
    nb_time_steps = 28
    dim_input_vector - 28
    input_shape = (nb_time_steps, dim_input_vector)
    model.add(LSTM(nb_lstm_outputs, input_shape=input_shape))
    model.add(Dense(nb_classes, activation='softmax', kernel_initializer=initializers.random_normal(stddev=0.01)))
    model.summary()
    plot_model(model, to_file='lstm_model.png')
    ```

## Sequential Model
## Specifying the input shape
## Compilation
## Training
## Functional API
## All models are callable, just like layers
## Multi-input and multi-output models
## Shared layers
## The concept of layer "node"

# Tensorflow
- reference [链接](https://www.tensorflow.org/programmers_guide/)
- 中文tensorflow指南 https://www.tensorflow.org/guide/low_level_intro?hl=zh-cn

## 内容
- 构建计算图tf.Graph
- 运行计算图tf.Session

- 图
```
计算图是排列成一个图的一系列 TensorFlow 指令。图由两种类型的对象组成。

操作（简称“op”）：图的节点。操作描述了消耗和生成张量的计算。
张量：图的边。它们代表将流经图的值。大多数 TensorFlow 函数会返回 tf.Tensors。

重要提示：tf.Tensors 不具有值，它们只是计算图中元素的手柄
```

- TensorBoard
```
TensorFlow 提供了一个名为 TensorBoard 的实用程序。TensorBoard 的诸多功能之一是将计算图可视化。您只需要使用几个简单的命令就能轻松完成此操作。

首先将计算图保存为 TensorBoard 摘要文件，具体操作如下所示：

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

这将在当前目录中生成一个 event 文件，其名称格式如下：

events.out.tfevents.{timestamp}.{hostname}

现在，在新的终端中使用以下 shell 命令启动 TensorBoard：

tensorboard --logdir .

接下来，在您的浏览器中打开 TensorBoard 的图页面，您应该会看到与以下图形类似的图：

```

- 会话(Session)
```
要评估张量，需要实例化一个 tf.Session 对象（非正式名称为会话）。会话会封装 TensorFlow 运行时的状态，并运行 TensorFlow 操作。如果说 tf.Graph 像一个 .py 文件，那么 tf.Session 就像一个 python 可执行对象。

下面的代码会创建一个 tf.Session 对象，然后调用其 run 方法来评估我们在上文中创建的 total 张量：

sess = tf.Session()
print(sess.run(total))

当您使用 Session.run 请求输出节点时，TensorFlow 会回溯整个图，并流经提供了所请求的输出节点对应的输入值的所有节点。

您可以将多个张量传递给 tf.Session.run。run 方法以透明方式处理元组或字典的任何组合，如下例所示：
print(sess.run({'ab':(a, b), 'total':total}))
它返回的结果拥有相同的布局结构：
{'total': 7.0, 'ab': (3.0, 4.0)}

在调用 tf.Session.run 期间，任何 tf.Tensor 都只有单个值。例如，以下代码调用 tf.random_uniform 来生成一个 tf.Tensor，后者会生成随机的三元素矢量（值位于 [0,1) 区间内）：

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))

每次调用 run 时，结果都会显示不同的随机值，但在单个 run 期间（out1 和 out2 接收到相同的随机输入值），结果显示的值是一致的：
[ 0.52917576  0.64076328  0.68353939]
[ 0.66192627  0.89126778  0.06254101]
(
  array([ 1.88408756,  1.87149239,  1.84057522], dtype=float32),
  array([ 2.88408756,  2.87149239,  2.84057522], dtype=float32)
)

部分 TensorFlow 函数会返回 tf.Operations，而不是 tf.Tensors。对指令调用 run 的结果是 None。您运行指令是为了产生副作用，而不是为了检索一个值。这方面的例子包括稍后将演示的初始化和训练操作。

```
- 供给
```
目前来讲，这个图不是特别有趣，因为它总是生成一个常量结果。图可以参数化以便接受外部输入，也称为占位符。占位符表示承诺在稍后提供值，它就像函数参数。

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

前面三行有点像函数。我们定义了这个函数的两个输入参数（x 和 y），然后对它们运行指令。我们可以使用 run 方法的 feed_dict 参数为占位符提供具体的值，从而评估这个具有多个输入的图：

print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

上述操作的结果是输出以下内容：

7.5
[ 3.  7.]

另请注意，feed_dict 参数可用于覆盖图中的任何张量。占位符和其他 tf.Tensors 的唯一不同之处在于如果没有为占位符提供值，那么占位符会抛出错误。
```

- 数据集
```
占位符适用于简单的实验，而数据集是将数据流式传输到模型的首选方法。

要从数据集中获取可运行的 tf.Tensor，您必须先将其转换成 tf.data.Iterator，然后调用迭代器的 get_next 方法。

创建迭代器的最简单的方式是采用 make_one_shot_iterator 方法。例如，在下面的代码中，next_item 张量将在每次 run 调用时从 my_data 阵列返回一行：

my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

到达数据流末端时，Dataset 会抛出 OutOfRangeError。例如，下面的代码会一直读取 next_item，直到没有数据可读：

while True:
  try:
    print(sess.run(next_item))
  except tf.errors.OutOfRangeError:
    break

如果 Dataset 依赖于有状态操作，则可能需要在使用迭代器之前先初始化它，如下所示：

r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    break

要详细了解数据集和迭代器，请参阅导入数据。
```

- 层
```
可训练的模型必须修改图中的值，以便在输入相同值的情况下获得新的输出值。将可训练参数添加到图中的首选方法是层。

层将变量和作用于它们的操作打包在一起。例如，密集连接层会对每个输出对应的所有输入执行加权和，并应用激活函数（可选）。连接权重和偏差由层对象管理。

创建层
下面的代码会创建一个 Dense 层，该层会接受一批输入矢量，并为每个矢量生成一个输出值。要将层应用于输入值，请将该层当做函数来调用。例如：

x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)

层会检查其输入数据，以确定其内部变量的大小。因此，我们必须在这里设置 x 占位符的形状，以便层构建正确大小的权重矩阵。

我们现在已经定义了输出值 y 的计算，在我们运行计算之前，还需要处理一个细节。

初始化层
层包含的变量必须先初始化，然后才能使用。尽管可以单独初始化各个变量，但也可以轻松地初始化一个 TensorFlow 图中的所有变量（如下所示）：

init = tf.global_variables_initializer()
sess.run(init)

重要提示：调用 tf.global_variables_initializer 仅会创建并返回 TensorFlow 操作的句柄。当我们使用 tf.Session.run 运行该操作时，该操作将初始化所有全局变量。

层函数的快捷方式
对于每个层类（如 tf.layers.Dense)，TensorFlow 还提供了一个快捷函数（如 tf.layers.dense）。两者唯一的区别是快捷函数版本是在单次调用中创建和运行层。例如，以下代码等同于较早的版本：
```

- 特征列
```
使用特征列进行实验的最简单方法是使用 tf.feature_column.input_layer 函数。此函数只接受密集列作为输入，因此要查看类别列的结果，您必须将其封装在 tf.feature_column.indicator_column 中。例如：

features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)

运行 inputs 张量会将 features 解析为一批向量。

特征列和层一样具有内部状态，因此通常需要将它们初始化。类别列会在内部使用对照表，而这些表需要单独的初始化操作 tf.tables_initializer。

var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))

初始化内部状态后，您可以运行 inputs（像运行任何其他 tf.Tensor 一样）：

print(sess.run(inputs))

这显示了特征列如何打包输入矢量，并将独热“department”作为第一和第二个索引，将“sales”作为第三个索引。

[[  1.   0.   5.]
 [  1.   0.  10.]
 [  0.   1.   8.]
 [  0.   1.   9.]]
```

- 训练
```
您现在已经了解 TensorFlow 核心部分的基础知识了，我们来手动训练一个小型回归模型吧。

定义数据
我们首先来定义一些输入值 x，以及每个输入值的预期输出值 y_true：

x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

定义模型
接下来，建立一个简单的线性模型，其输出值只有 1 个：

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)

您可以如下评估预测值：

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))

该模型尚未接受训练，因此四个“预测”值并不理想。以下是我们得到的结果，您自己的输出应该有所不同：

[[ 0.02631879]
 [ 0.05263758]
 [ 0.07895637]
 [ 0.10527515]]

损失
要优化模型，您首先需要定义损失。我们将使用均方误差，这是回归问题的标准损失。

虽然您可以使用较低级别的数学运算手动定义，但 tf.losses 模块提供了一系列常用的损失函数。您可以使用它来计算均方误差，具体操作如下所示：

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

print(sess.run(loss))

这会生成如下所示的一个损失值：

2.23962

训练
TensorFlow 提供了执行标准优化算法的优化器。这些优化器被实现为 tf.train.Optimizer 的子类。它们会逐渐改变每个变量，以便将损失最小化。最简单的优化算法是梯度下降法，由 tf.train.GradientDescentOptimizer 实现。它会根据损失相对于变量的导数大小来修改各个变量。例如：

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

该代码构建了优化所需的所有图组件，并返回一个训练指令。该训练指令在运行时会更新图中的变量。您可以按以下方式运行该指令：

for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

由于 train 是一个指令而不是张量，因此它在运行时不会返回一个值。为了查看训练期间损失的进展，我们会同时运行损失张量，生成如下所示的输出值：

1.35659
1.00412
0.759167
0.588829
0.470264
0.387626
0.329918
0.289511
0.261112
0.241046
...

完整程序
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

print(sess.run(y_pred))
```

- 什么是 tf.Graph？

- minibatch training
```
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


def load_mnist(mnist_path='MNIST_data/'):
    """
    Load mnist data from tensorflow.
    :param mnist_path: Path where the mnist data will be stored
    :return:
    mnist -- Tensorflow handwriting data
        train - Flattened and normalized images of shape (55000, 784), width and height of a image is 28
        test - Flattened and normalized images of shape (10000, 784), width and height of a image is 28
        (train, test).label - One hot encoded labels of shape (n, 10), n is number of data
    """
    mnist = input_data.read_data_sets(mnist_path, one_hot=True)

    print('Number of training examples: ' + str(mnist.train.num_examples))
    print('Number of test examples: ' + str(mnist.test.num_examples))

    return mnist


def initialize_parameters(num_px, num_class):
    """
    Initialize parameters to build NN.
        w1: (num_l1, num_px)
        b1: (num_l1, 1)
        w2: (num_l2, num_l1)
        b2: (num_l2, 1)
        w3: (num_class, num_l2)
        b2: (num_class, 1)
    :param num_px: Number of pixels of a image, 784
    :param num_class: Number of class, 10
    :return:
    parameters -- A python dictionary of tensors
    """
    num_l1 = 512
    num_l2 = 512

    w1 = tf.get_variable('W1', [num_l1, num_px], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [num_l1, 1], initializer=tf.zeros_initializer())
    w2 = tf.get_variable('W2', [num_l2, num_l1], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [num_l2, 1], initializer=tf.zeros_initializer())
    w3 = tf.get_variable('W3', [num_class, num_l2], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', [num_class, 1], initializer=tf.zeros_initializer())

    parameters = {'W1': w1,
                  'b1': b1,
                  'W2': w2,
                  'b2': b2,
                  'W3': w3,
                  'b3': b3}

    return parameters


def forward_propagation(x, parameters, keep_prob):
    """
    Implement of forward propagation of the following model.
        LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFT MAX
    :param x: Placeholder of input data set
    :param parameters: Python dictionary containing W, b
    :param keep_prob: Probability of keeping a neuron active during drop-out
    :return:
    z3 -- The output value before soft max activation function
    """
    w1 = parameters['W1']
    b1 = parameters['b1']
    w2 = parameters['W2']
    b2 = parameters['b2']
    w3 = parameters['W3']
    b3 = parameters['b3']

    z1 = tf.add(tf.matmul(w1, x), b1)
    a1 = tf.nn.dropout(tf.nn.relu(z1), keep_prob)
    z2 = tf.add(tf.matmul(w2, a1), b2)
    a2 = tf.nn.dropout(tf.nn.relu(z2), keep_prob)
    z3 = tf.add(tf.matmul(w3, a2), b3)

    return z3


def compute_cost(z3, y, parameters):
    """
    Compute the cost.
    :param z3: Output of forward propagation
    :param y: Labels
    :param parameters: Python dictionary containing W, b
    :return:
    cost -- A Tensor of the cross entropy cost function
    """
    # Calc L2 loss
    lambd = 0.
    w1 = parameters['W1']
    w2 = parameters['W2']
    w3 = parameters['W3']

    regularize = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)

    logits = tf.transpose(z3)
    labels = tf.transpose(y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels) +
                          lambd * regularize)
    return cost


def model(mnist, learning_rate=0.0002, num_epochs=100, mini_batch_size=128, print_cost=True):
    """
    Implement of 3-layer tensorflow model.
    :param mnist: Tensorflow handwriting data
    :param learning_rate: Learning rate of gradient descent
    :param num_epochs: Number of iteration epoch loop
    :param mini_batch_size: Size of mini batch
    :param print_cost: If true, print the cost
    :return:
    parameters -- Parameters learnt by this model
    """
    # Get transpose of data, because I want one column to represent one datum
    train_x = mnist.train.images.T
    train_y = mnist.train.labels.T
    test_x = mnist.test.images.T
    test_y = mnist.test.labels.T

    num_px = train_x.shape[0]  # number of pixels (26 * 28 = 784)
    num_class = train_y.shape[0]  # number of class should be 10

    # Create placeholders
    x = tf.placeholder(tf.float32, [num_px, None])
    y = tf.placeholder(tf.float32, [num_class, None])
    keep_prob = tf.placeholder(tf.float32)

    # Initialize parameters
    parameters = initialize_parameters(num_px, num_class)

    # Forward propagation
    z3 = forward_propagation(x, parameters, keep_prob)

    # Cost function
    cost = compute_cost(z3, y, parameters)

    # Back propagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize tensorflow variables
    init = tf.global_variables_initializer()

    costs = []
    with tf.Session() as session:
        session.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_mini_batches = mnist.train.num_examples // mini_batch_size

            for _ in range(num_mini_batches):
                mini_batch_x, mini_batch_y = mnist.train.next_batch(mini_batch_size)
                _, mini_batch_cost = session.run([optimizer, cost], feed_dict={x: mini_batch_x.T,
                                                                               y: mini_batch_y.T,
                                                                               keep_prob: 0.8})
                epoch_cost += mini_batch_cost / num_mini_batches

            if print_cost and epoch % 10 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost and epoch % 5 == 0:
                costs.append(epoch_cost)

        # Plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = session.run(parameters)

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(z3), tf.argmax(y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({x: train_x, y: train_y, keep_prob: 1}))
        print("Test Accuracy:", accuracy.eval({x: test_x, y: test_y, keep_prob: 1}))

        return parameters


def main():
    mnist = load_mnist()
    model(mnist)


main()
```

## Estimator



## Eager Execution
- 需要最新的tensorflow
- 如何启用eager模式 tf.enable_eager_execution()
- 检查eager状态 tf.executing_eagerly()
- 动态控制流，python的if
- 使用tf.keras的框架来写训练程序，注意整个程序的写法 https://www.tensorflow.org/guide/eager
- tfe=tf.contrib.eager
- 如何在keras框架下自己写训练循环
- tfe.metrics

## Tensorflow APIs
### 变量
```
variable = tf.Variable(default, name=name, trainable=False) # 注意trainable可以指定变量为不可训练
placeholder = tf.placeholder(dtype=variable.dtype,
                             shape=variable.get_shape(),
                             name=(name + "/placeholder")) # 产生一个place holder型tensor, 执行时需要喂值给它


tf.variable_scope可以让不同命名空间中的变量取相同的名字，无论tf.get_variable或者tf.Variable生成的变量
tf.name_scope具有类似的功能，但只限于tf.Variable生成的变量

tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope.name)

```
### 命名
- tf.add_to_collection(name, single_variable)


### 变量转换
- tf.convert_to_tensor(...)

### 训练
- tf.train
  - tf.train.latest_checkpoint(path)
  - tf.train.Checkpoint(optimizer, model, optimizer_step)
  - tf.train.get_or_create_global_step()
  - tf.train.Checkpoint(x=tf_variable).save(path)
  - tf.train.GradientDesceneOptimizer(leraning_rate=0.01)
  - tf.train.ExponentialMovingAverage
- 

### Estimator
- tf.estimator.Estimator
- tf.keras.estimator.model_to_estimator
- tf.feature_column
- tf.estimator.LinearClassifier
-tf.estimator.Estimator.train

# sklean
## sklearn APIs
- sklearn.base
- sklearn.cluster
- sklearn.neighbors
- sklearn.compose
- sklearn.covariance
- sklearn.datasets
- sklearn.decomposition
- sklearn.discriminant_analysis
- sklearn.ensemble
- sklearn.pipeline
- sklearn.preprocessing
- sklearn.semi_supervised
- sklearn.svm
  - sklearn.svm.libsvm
- sklearn.tree
- sklearn.utils
- sklearn.exceptions
- sklearn.feature_extraction
  - sklearn.feature_extraction.image
  - sklearn.feature_extraction.text
- sklearn.feature_selection
- skleran.gaussian_process
  - sklearn.gaussian_process.kernels
- sklearn.linear_model
- sklearn.manifold
- sklearn.metrics
  - sklearn.metrics.pairwise
- sklearn.mixture
- sklearn.neural_network
- sklearn.model_selection
- sklearn.kernel_approximation
- sklearn.kernel_ridge
- sklearn.multiclass
- sklearn.multioutput
- sklearn.naive_bayes
- sklearn.cross_decomposition 
- sklearn.calibration
- sklearn.random_projection
- sklearn.dummy
- sklearn.isotonic
- sklearn.impute


## 量化

- 量化Quantization
  - 使用场景 : gif图片压缩。颜色用颜色画板color palette给出，每个像素只要存color palette中的index即可。
  - RGB彩色图：每个像素需要24位。
  - Vector Quantization：每个像素只要8位。
  - KMeans给出的n-colors聚类中心作为codebook。

  ```
  
  ```

## 多标签
- sklearn.preprocessing.MultiLabelBinarier
  return a binary matrix indicating the presence of a class label.
  ```
  >>> from sklearn.preprocessing import MultiLabelBinarizer
  >>> mlb = MultiLabelBinarizer()
  >>> mlb.fit_transform([(1, 2), (3,)])
  array([[1, 1, 0],
         [0, 0, 1]])
  >>> mlb.classes_
  array([1, 2, 3])
  >>>
  >>> mlb.fit_transform([set(['sci-fi', 'thriller']), set(['comedy'])])
  array([[0, 1, 1],
         [1, 0, 0]])
  >>> list(mlb.classes_)
  ['comedy', 'sci-fi', 'thriller']
  ```

## 数据预处理
- LOF: 根据kdtree或者聚类方法把outlier去掉。
