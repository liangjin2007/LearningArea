# userguide https://numpy.org/doc/stable/user/index.html

## 基础介绍
import numpy as np
```
数值python库

多维数组数据结构ndarray

操作这种多维数组的函数库

非常快
```

初始化及一些slicing:
```
所有元素必须是相同类型

创建完后它的尺寸是不可改变的？

从list初始化， a = np.array([1, 2, 3, 4, 5]); 从nested list初始化

0-based integeindexing: a[0]

可以像list一样slicing: a[:3],  但是有个区别的地方是numpy返回的是一个叫view的东西，是重用了a中的数据。 这个在pytorch中也有这个view的概念。
```

数组属性：
```
a.ndim
a.shape
a.size
a.dtype
```

创建函数
```
np.ones, np.zeros
np.arange(4)
```

