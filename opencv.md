## OpenCV学习

- MSER检测(最大稳定极值区域)![mser](https://github.com/liangjin2007/data_liangjin/blob/master/opencv_mser.jpg?raw=true)
  - 仿射不变性
  - 区域增长
  - 类似于分水岭算法
  - 多级灰度threshold

- Blob检测 用于几何形状提取和分析
  - 圆度
  - 

- Edge Detection
  - 求偏导数
    - Scharr
    - Sobel
  - Canny Detection
    - threshold
  
- dft
  "An Intuitive Explanation of Fourier Theory"
  - 离散傅立叶变换
    - 任何信号（我们的case是可视图像）都可以表示成一系列sinusoids函数的和。
    - 输入信号包含三个部分： 空间频率frequency，振幅magnitude，相位phase
    - 从简单到复杂地去理解dft:
        - 1D Fourier Transform: 如果输入是一个sinusoid, 那么输出是一个单个的peak at point f。
        - 2D Fourier Transform: 输出图像中 横轴为原来图像中沿着x方向的频率， 纵轴为原图中沿着y方向的频率， 亮度为原图中的亮度对比度。
    - DC Term:对应于零频率，代表平均亮度
    - nyquist frequency

- facial_features
  可以指定眼睛等位置

- facedetect
  脸部检测
  
- pyramids
  生成图像的金字塔
  
- convexhull
  算一堆点的凸包
  
- morphology
腐蚀/膨胀/开/闭
开运算：先
    
 
    


