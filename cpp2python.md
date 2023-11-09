# 给pytorch支持一个用c++ + cuda写的forward/backward函数

xxx.cpp
```
#include <torch/torch.h>


```

xxx.cu


xxx.py
```
cd = load(name="cd", sources=["xxx.cpp", "xxx.cu"]) 

```




