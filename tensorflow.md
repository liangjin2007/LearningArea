# Tensorflow学习
- reference [链接](https://www.tensorflow.org/programmers_guide/)
- 中文tensorflow指南 https://www.tensorflow.org/guide/low_level_intro?hl=zh-cn

## 内容
- 构建计算图tf.Graph
- 运行计算图tf.Session
```
计算图是排列成一个图的一系列 TensorFlow 指令。图由两种类型的对象组成。

操作（简称“op”）：图的节点。操作描述了消耗和生成张量的计算。
张量：图的边。它们代表将流经图的值。大多数 TensorFlow 函数会返回 tf.Tensors。

重要提示：tf.Tensors 不具有值，它们只是计算图中元素的手柄
```
- 使用TensorBoard
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
- 
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

