# sklearn APIs
- sklearn.base
- sklearn.calibration
- sklearn.cluster
- sklearn.compose
- sklearn.covariance
- sklearn.cross_decomposition 

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

