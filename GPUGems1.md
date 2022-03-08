### 第一章 水模拟
- Siggraph 2001 Tessendorf Simulating Ocean Water https://cupdf.com/document/simulating-ocean-water.html
```
4个正弦函数叠加，A_i x cos(dot(D_i(x, y), (x, y)) x omega_i + t x phi_i)
  角频率 omega_i = 2Pi / L, L指波长； phi_i表示波在每秒移动的距离S * omega_i； A_i振幅； D_i为垂直于波阵面的水平向量。
```

### 第二章 蚀刻模拟
```
假设形成蚀刻的光都是垂直地从海底射出的。
```

### 第三章 Dawn皮肤
![皮肤](https://github.com/liangjin2007/data_liangjin/blob/master/Dawn1.jpg?raw=true)

### 第四章 Dawn动画
![动画](https://github.com/liangjin2007/data_liangjin/blob/master/%E5%8A%A8%E7%94%BB.jpg?raw=true)

### 第五章 改进的Perlin Noise
- Perlin Noise
![Perlin NOise](https://github.com/liangjin2007/data_liangjin/blob/master/PerlinNoise.jpg?raw=true)

- 改进
```
问题：
1.二阶导数含有非0值
2.结果中有高频
```
![改进的Perlin](https://github.com/liangjin2007/data_liangjin/blob/master/%E6%94%B9%E8%BF%9B%E7%9A%84Perline.jpg?raw=true)

