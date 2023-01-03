## [第一部分](https://www.zhihu.com/topic/20016366/hot)

- 测试（样本）x1, x2都是随机变量有一定的变化范围 x1 +- 5, x2 +- 9, 如何得到两次测试（样本）的最优估计？
```
x^ = w1 x1 + w2 x2
w1 + w2 = 1
x1的方差为d1
x2的方差为d2
```

- 上面两式得到x^的方差
```
d^2 = w1^2 d1^2 + w2^2 d2^2
    = (1-w)^2 d1^2 + w^2 d2^2
```

- 最优估计, 求使得d^2最小的w
```
最优
w = d1^2/(d1^2 + d2^2)

x = d1^2/(d1^2 + d2^2) x1 + d2^2/(d1^2 + d2^2) x2

d^2 = d1^2 d2^2 / (d1^2 + d2^2)
```

- 运动估计的 g-h-k滤波器
```
状态外推方程 或者 Predict预测函数:

    x_pred = hat(x) + hat(v) dt + 0.5 hat(a) dt^2
    v_pred = hat(v) + hat(a) dt
    a_pred = hat(a) + f(t, dt)
    hat(xxx)表示估计值。

状态更新方程 Update函数:     得到预测值和测量值， 将预测值和测量值进行加权平均，得到

    hat(x) = x_pred + g(xz - x_pred)
    hat(v) = v_pred + h((xz - x_pred)/dt)
    hat(a) = a_pred + k((xz - x_pred/(0.5*dt^2))

g, h, k要根据实际情况手工调整。
```

