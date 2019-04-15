# Tensorflow学习
- reference [链接](https://www.tensorflow.org/programmers_guide/)
- Eager Execution
  - 使用tf.keras的框架来写训练程序，注意整个程序的写法 https://www.tensorflow.org/guide/eager
  ```
  ...
  for (batch, (images, labels)) in enumerate(dataset.take(400)):
  if batch % 80 == 0:
    print()
  print('.', end='')
  with tf.GradientTape() as tape:
    logits = mnist_model(images, training=True)
    loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

  loss_history.append(loss_value.numpy())
  grads = tape.gradient(loss_value, mnist_model.variables)
  optimizer.apply_gradients(zip(grads, mnist_model.variables),
                            global_step=tf.train.get_or_create_global_step())
  ...
  ```
- 
- 

