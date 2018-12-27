# 数学原理
- Jensen不等式 
  - 如果f是凸函数，X是随机变量，那么E[f(X)] >= f(E[X])
  
- EM
  - [博客](http://www.cnblogs.com/jerrylead/archive/2011/04/06/2006936.html)


# 度量方面

- Earth Mover's Distance(EMD)
  - [github](https://github.com/wmayner/pyemd)

- MSE

- PSNR
  - 10*log10(square(L)/MSE)

- Structural Similarity (SSIM) Index in Image Space(SSIM)
  - tf.image.ssim
  - [skimage](http://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html)
  - [github](https://github.com/jterrace/pyssim)
  - f(l(x,y)*c(x,y)*s(x,y))

- Luminance
  - mean intensity : l(x) = mean(x) = sum(x)/N

- move to center
  - x-l(x) means project x onto the hyperplane of sum(x) = 0.
  
- Contrast
  - standard deviation(root of squared variance) as contrast. Use unbiased estimation 
  - contrast(x) = sqrt(sum(square(x-l(x)))/(N-1))
  
- normalize
  - (x-l(x))/contrast(x)

- l(x,y) range in 0~1 and avoid numeric overflow
  - (2*l(x)*l(y)+C)/(square(l(x))+square(l(y))+C

- structure comparison
  - correlation(inner product) correlation(x, y) = sum((xi-l(x))*(yi-l(y))/(N-1)
  - s(x,y) = (correlation(x, y)+C)/((contrast(x)*contrast(y))+C)


# 算法/方法方面
- [2017][ICCV]Deep Adaptive Image Clustering.pdf
    逐对训练

- 


