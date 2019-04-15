# sklearn APIs
- sklearn.base
- sklearn.cluster
- sklearn.neighbors
- sklearn.compose
- sklearn.covariance
- sklearn.datasets
- sklearn.decomposition
- sklearn.discriminant_analysis
- sklearn.ensemble
- sklearn.pipeline
- sklearn.preprocessing
- sklearn.semi_supervised
- sklearn.svm
  - sklearn.svm.libsvm
- sklearn.tree
- sklearn.utils
- sklearn.exceptions
- sklearn.feature_extraction
  - sklearn.feature_extraction.image
  - sklearn.feature_extraction.text
- sklearn.feature_selection
- skleran.gaussian_process
  - sklearn.gaussian_process.kernels
- sklearn.linear_model
- sklearn.manifold
- sklearn.metrics
  - sklearn.metrics.pairwise
- sklearn.mixture
- sklearn.neural_network
- sklearn.model_selection
- sklearn.kernel_approximation
- sklearn.kernel_ridge
- sklearn.multiclass
- sklearn.multioutput
- sklearn.naive_bayes
- sklearn.cross_decomposition 
- sklearn.calibration
- sklearn.random_projection
- sklearn.dummy
- sklearn.isotonic
- sklearn.impute


# 量化

- 量化Quantization
  - 使用场景 : gif图片压缩。颜色用颜色画板color palette给出，每个像素只要存color palette中的index即可。
  - RGB彩色图：每个像素需要24位。
  - Vector Quantization：每个像素只要8位。
  - KMeans给出的n-colors聚类中心作为codebook。

  ```
  
  ```
- sklearn.preprocessing.MultiLabelBinarier
return a binary matrix indicating the presence of a class label.
```
>>> from sklearn.preprocessing import MultiLabelBinarizer
>>> mlb = MultiLabelBinarizer()
>>> mlb.fit_transform([(1, 2), (3,)])
array([[1, 1, 0],
       [0, 0, 1]])
>>> mlb.classes_
array([1, 2, 3])
>>>
>>> mlb.fit_transform([set(['sci-fi', 'thriller']), set(['comedy'])])
array([[0, 1, 1],
       [1, 0, 0]])
>>> list(mlb.classes_)
['comedy', 'sci-fi', 'thriller']
```

