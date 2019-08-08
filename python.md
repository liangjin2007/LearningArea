# generator
- yield

# 上下文管理器
- with
- contextmanager装饰器
- https://www.cnblogs.com/zhbzz2007/p/6158125.html


# ndarray.sort
- [ndarray.sort(order='y')](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ndarray.sort.html#numpy.ndarray.sort)

# list.sort
a = [("a",0.1),("b",0.05),("c",0.06)]

a.sort(key=lambda x:x[1], reverse=True)

# 取numpy数组中某一个axis
a[0,:] # 取第一行

a[:,:,1] # 2维

a[:,:,0:1] # 3维

# 取字符串子串
a="abcc_def"
a[5:] # "def"

# make tuple longer
a = (None,)+(3,5) # (None, 3, 5)

# 使用字典作为某函数的扩展参数
options = {
    "activation": "relu",
    "kernel_size": (3, 3),
    "padding": "same"
}

function_A(**options)

# method
- Python 语法
@property
@staticmethod

# 数组操作
np.round
np.reshape(a, [-1])
c, d = np.meshgrid([1,2,3,4], [4,5,6,7]) #
[[1 2 3 4]
 [1 2 3 4]
 [1 2 3 4]
 [1 2 3 4]]
[[4 4 4 4]
 [5 5 5 5]
 [6 6 6 6]
 [7 7 7 7]]
 e = np.stack([c,d,c,d], axis=0)
 [[1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]
 [4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7]
 [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]
 [4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7]]
 
# datatime
datetime在python中比较常用，主要用来处理时间日期，使用前先倒入datetime模块。下面总结下本人想到的几个常用功能。
```
print datetime.datetime.now()
datetime.timedelta(days=1)
(datetime.datetime.now() - datetime.datetime.utcnow()).total_seconds()

datetime转str格式：
datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
str格式转datetime格式：
datetime.datetime.strptime("2015-07-17 16:58:46","%Y-%m-%d %H:%M:%S")
```
# oss2 api
https://github.com/aliyun/aliyun-oss-python-sdk/blob/master/oss2/api.py

# slicing ::
```
这个是python的slice notation的特殊用法。

a = [0,1,2,3,4,5,6,7,8,9]
b = a[i:j] 表示复制a[i]到a[j-1]，以生成新的list对象
b = a[1:3] 那么，b的内容是 [1,2]
当i缺省时，默认为0，即 a[:3]相当于 a[0:3]
当j缺省时，默认为len(alist), 即a[1:]相当于a[1:10]
当i,j都缺省时，a[:]就相当于完整复制一份a了

b = a[i:j:s]这种格式呢，i,j与上面的一样，但s表示步进，缺省为1.
所以a[i:j:1]相当于a[i:j]
当s<0时，i缺省时，默认为-1. j缺省时，默认为-len(a)-1
所以a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍。所以你看到一个倒序的东东。
```
# numpy数组二维索引取子矩阵
```
>>> a = np.arange(12).reshape(3,4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> i = np.array( [ [0,1],                      # 第一个轴的索引
...                 [1,2] ] )
>>> j = np.array( [ [2,1],                        # 第二个轴的索引
...                 [3,3] ] )
>>>
>>> a[i,j]                                     # i 和 j形状必须相同
array([[ 2,  5],
       [ 7, 11]])
```
