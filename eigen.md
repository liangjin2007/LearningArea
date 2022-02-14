### Eigen使用快速参考 http://eigen.tuxfamily.org/dox/AsciiQuickReference.txt 包含跟Matlab代码的对比

### 其他

- MatrixXd to VectorXd
```
// Column by column
MatrixXd A(3,2);
A << 1,2,3,4,5,6;
VectorXd B(Map<VectorXd>(A.data(), A.cols()*A.rows()));

// row by row
MatrixXd A(3,2);
A << 1,2,3,4,5,6;
A.transposeInPlace();
VectorXd B(Map<VectorXd>(A.data(), A.cols()*A.rows()));
```


