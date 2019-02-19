
# matplotlib
import matplotlib.pyplot as plt

2行，10列，无轴，灰度色
====================================================
```
# 显示结果
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    # 分成2*n个子图，占用第i+1个子图
    ax = plt.subplot(2, n, i + 1)                     # 这里画第是第1行
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    # x轴，y轴不可见
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    # 分成2*n个子图，占用第i+1+n个子图
    ax = plt.subplot(2, n, i + 1 + n)                 # 注意这里在画第2行
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
