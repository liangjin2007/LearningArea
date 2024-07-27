# userguide https://numpy.org/doc/stable/user/index.html

## 基础介绍
import numpy as np

基本概念：
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

创建函数:
```
np.ones, np.zeros
np.arange(4)
```

添加，删除，排序:
```
np.sort(a)
np.argsort
np.
```

Reshape:
```
np.reshape(a, newshape=(3, 2), order='C')
a.reshape((3, 2))
```

New axis, 1D to 2D:
```
np.newaxis, np.expand_dims
a2 = a[np.newaxis, :] # (1, 6) 行向量
a3 = a[:, np.newaxis] # (6, 1) 列向量

b = np.expand_dims(a, axis= 1)  # (6, 1) 
```

Indexing and Slicing：
```
a[-2:]  # 从倒数第二个数开始
```

Selection：
```
a[a < 5] # 返回的是一个列表
选择偶数：
divisible_by_2 = a[a % 2 == 0]

```

Create array from existing data(arrays)
```
slicing and indexing, np.vstack(), np.hstack(), np.hsplit(), .view(), copy()

```


数组基本操作：
```
数学运算 + - * /
a.sum()
a.sum(axis= 1)
a.max()
a.min

```

Broadcasting：
```

```

创建矩阵：
```
```

创建随机数
```
Generator.integers
```

