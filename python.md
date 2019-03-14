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

# datatime
datetime在python中比较常用，主要用来处理时间日期，使用前先倒入datetime模块。下面总结下本人想到的几个常用功能。
```
1、当前时间：
>>> print datetime.datetime.now()
2015-07-17 16:39:15.712000
>>> print type(datetime.datetime.now())
<type 'datetime.datetime'>
返回的datetime时间格式。
2、当前日期
>>> print datetime.datetime.now().date()
2015-07-17
>>> print type(datetime.datetime.now().date())
<type 'datetime.date'>
3、当前时间tuple
>>> datetime.datetime.now().timetuple()
time.struct_time(tm_year=2015, tm_mon=7, tm_mday=17, tm_hour=16, tm_min=51, tm_sec=26, tm_wday=4, tm_yday=198, tm_isdst=-1)
>>> datetime.datetime.now().timetuple().tm_mday
17
4、时间移动（几天、几小时前后...）
使用datetime.timedelta这个方法来前后移动时间，可以用的参数有weeks，days，hours，minutes，seconds，microseconds。

>>> print datetime.datetime.now() + datetime.timedelta(days=1)
2015-07-18 16:49:48.574000
>>> print datetime.datetime.now() + datetime.timedelta(hours=1)
2015-07-17 17:49:57.122000
>>> print datetime.datetime.now() + datetime.timedelta(minutes=-30)
2015-07-17 16:20:08.619000
上个月最后一天

>>> print datetime.date(day=1,month=datetime.date.today().month,year=datetime.date.today().year) - datetime.timedelta(days=1)
2015-06-30
5、获取两个时间的时间差
>>> (datetime.datetime.now() - datetime.datetime.utcnow()).total_seconds()
28800.0
6、时间转化
datetime转str格式：
>>> datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
'2015-07-17 16:58:46'
str格式转datetime格式：
>>> datetime.datetime.strptime("2015-07-17 16:58:46","%Y-%m-%d %H:%M:%S")
datetime.datetime(2015, 7, 17, 16, 58, 46)
>>> print datetime.datetime.strptime("2015-07-17 16:58:46","%Y-%m-%d %H:%M:%S")
2015-07-17 16:58:46
>>> print type(datetime.datetime.strptime("2015-07-17 16:58:46","%Y-%m-%d %H:%M:%S"))
<type 'datetime.datetime'>
datetime转timestamp：
>>> import time
>>> now=datetime.datetime.now()
>>> time.mktime(now.timetuple())
1437123812.0
timestamp转datetime:
>>> datetime.datetime.fromtimestamp(1437123812.0)
datetime.datetime(2015, 7, 17, 17, 3, 32)
>>> print datetime.datetime.fromtimestamp(1437123812.0)
2015-07-17 17:03:32
```

