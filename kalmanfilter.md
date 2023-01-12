## [第一部分](https://www.zhihu.com/topic/20016366/hot)

- **混合**：测试（样本）x1, x2都是随机变量有一定的变化范围 x1 +- 5, x2 +- 9, 如何得到两次测试（样本）的最优估计？
```
x^ = w1 x1 + w2 x2
w1 + w2 = 1
x1的方差为d1
x2的方差为d2

上面两式得到x^的方差
d^2 = w1^2 d1^2 + w2^2 d2^2
    = (1-w)^2 d1^2 + w^2 d2^2
```

- **最优估计**, 求使得d^2最小的w
```
最优
w = d1^2/(d1^2 + d2^2)

x = d1^2/(d1^2 + d2^2) x1 + d2^2/(d1^2 + d2^2) x2

d^2 = d1^2 d2^2 / (d1^2 + d2^2)
```

- **运动估计的 g-h-k滤波器**
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

g, h, k要根据实际情况手工调整。因为没有引入概率，没有得到最优估计的g,h,k.
```

- Beyes滤波器

- **Kalman滤波器**
```
在g-h-k滤波器的基础上引入了概率，从而得到了最优状态估计。

Predict和Update中的公式可以在很多地方找到，比如opencv代码（kalmanfilter.cpp）。

在没有推导理解好公式之前，可以看opencv的代码和这个链接 https://thinkautonomous.medium.com/computer-vision-for-tracking-8220759eee85

涉及到几块内容 ： 
1. x' = A x, x可以是(cx, cy, w, h, vx, vy, vw vh)这样的状态变量，我也实践过(x, y, z, vx, vy, vz, ax, ay, az)这样的状态变量
2. 关于H和测量量：测量量可以是只有cx, cy, w, h, 或者 我实践过的我只用了(x, y, z)作为测量量。此时H是非方阵的单位阵。 
3. 关于P, Q， R， 统计协方差相关的几个量， 需要预设。 上面链接中有做适当解释。
4. Predict 和 Update: Update中有涉及到矩阵求逆，像我实践过的case, 是一个3x3矩阵求逆。  K卡尔曼增益矩阵是个N x 3的矩阵。 




```

## [第二部分](https://github.com/kcg2015/Vehicle-Detection-and-Tracking)

