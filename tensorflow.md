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

