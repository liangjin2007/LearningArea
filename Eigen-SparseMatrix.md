## 存储结构
```
Eigen的SparseMatrix采用**压缩稀疏列（Compressed Sparse Column, CSC）**格式作为默认存储方式（也支持压缩稀疏行CSR格式），这是数值计算领域最常用的稀疏矩阵存储格式之一，平衡了存储效率与计算性能。
CSR: Row Major Sparse Matrix Eigen::SparseMatrix<double, Eigen::RowMajor>
  values	std::vector<T>	存储所有非零元素的值，按列优先顺序排列
  innerIndices	std::vector<int>	存储每个非零元素对应的列索引，与values数组一一对应
  outerStarts	std::vector<int>	存储每一列（CSC）或每一行（CSR）的起始位置，长度为cols()+1（CSC）

CSC: Col Major Sparse Matrix Eigen::SparseMatrix<double, Eigen::ColMajor>
  values	std::vector<T>	存储所有非零元素的值，按列优先顺序排列
  innerIndices	std::vector<int>	存储每个非零元素对应的行索引，与values数组一一对应
  outerStarts	std::vector<int>	存储每一列（CSC）或每一行（CSR）的起始位置即在values中的索引，长度为cols()+1（CSC）

```
