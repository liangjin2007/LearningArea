# Tensorflow学习
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

# name_scope and variable_scope
tf.name_scope is a context manager for use when defining a Python op.
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

