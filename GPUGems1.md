### 第一章 水模拟
- Siggraph 2001 Tessendorf Simulating Ocean Water https://cupdf.com/document/simulating-ocean-water.html
```
4个正弦函数叠加，A_i x cos(dot(D_i(x, y), (x, y)) x omega_i + t x phi_i)
  角频率 omega_i = 2Pi / L, L指波长； phi_i表示波在每秒移动的距离S * omega_i； A_i振幅； D_i为垂直于波阵面的水平向量。

几何波
纹理波

  
```
