# 数学原理
- Jensen不等式 
  - 如果f是凸函数，X是随机变量，那么E[f(X)] >= f(E[X])
  
- EM
  - [博客](http://www.cnblogs.com/jerrylead/archive/2011/04/06/2006936.html)
  - EM是一种解决存在隐含变量优化问题的有效方法
  - 目标：独立同分布样本{x(i)}, 每个样本隐含一个类别z, 要使得p(x, z)最大。 

- 
# 度量方面

- Earth Mover's Distance(EMD)
  - [github](https://github.com/wmayner/pyemd)

- MSE
  - keras
  ```
  return K.mean(K.square(y_pred - y_true), axis=-1)
  ```

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

- spearman ranking
  - 度量预测数组和true数组的排序情况。 这个度量在结合mse的情况下，可以用来度量，预测数组的相对排序情况。
  ```
  def spearman_rank2(y_true, y_pred):
    # tf.argsort(values)
    # tf.gather(values, tf.argsort(values))
    
    n = y_pred.shape[0]
    a = [0.0]*n
    b = [0.0]*n
    for i in range(n):
        a[i] = (i, y_pred[i])

    for i in range(n):
        b[i] = (i, y_true[i])
     
    c = np.array(a, dtype=[('x', int), ('y', float)])
    d = np.array(b, dtype=[('x', int), ('y', float)])

    c.sort(order='y')
    d.sort(order='y')

    for i,v in enumerate(c.tolist()):
        a[v[0]] = i
    for i,v in enumerate(d.tolist()):
        b[v[0]] = i
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)

    return 1.0 - (np.sum(np.square(a-b))*6.0/(n**2*n-n))
  ```

# 算法/方法方面
- [2017][ICCV]Deep Adaptive Image Clustering.pdf
    逐对训练

- 


