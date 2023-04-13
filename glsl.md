## 类c语法
- 注意事项
```
1. 无法使用&及指针

2. 每个顶点的属性的component是有上限的，最长16，代表16个字节，4个float长度。 比如我查了一下，发现是16。
int nrAttributes;
glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &nrAttributes);
std::cout << "Maximum nr of vertex attributes supported: " << nrAttributes << std::endl;
当我视图塞一个vec8的数组给顶点buffer的时候会失败。解决办法是再申请一个顶点buffer？ 或者压缩一下？


```

- 语法
```


```
