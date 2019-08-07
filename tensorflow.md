# Tensorflow学习
- reference [链接](https://www.tensorflow.org/programmers_guide/)
## Eager Execution
- 需要最新的tensorflow
- 如何启用eager模式 tf.enable_eager_execution()
- 检查eager状态 tf.executing_eagerly()
- 动态控制流，python的if
- 使用tf.keras的框架来写训练程序，注意整个程序的写法 https://www.tensorflow.org/guide/eager
- tfe=tf.contrib.eager
- 如何在keras框架下自己写训练循环
- tfe.metrics

# Tensorflow APIs
### 变量
```
variable = tf.Variable(default, name=name, trainable=False) # 注意trainable可以指定变量为不可训练
placeholder = tf.placeholder(dtype=variable.dtype,
                             shape=variable.get_shape(),
                             name=(name + "/placeholder")) # 产生一个place holder型tensor, 执行时需要喂值给它

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
- 

