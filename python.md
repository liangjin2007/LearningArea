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

