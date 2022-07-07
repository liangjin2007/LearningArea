## sphere traced ray marching https://github.com/sungiant/sdf
- 里面有图，关于sphere traced ray marching on sdf。比较好理解
## Sdf的交并差
```
Take the union of two SDFs by
taking the min() of their
functions.
● Take the intersection of two
SDFs by taking the max() of their
functions.
● The max() of function A and the
negative of function B will return
the difference of A - B.

```
## marching cube
- 前几年看过这方面的论文，论文都比较早期，paper涉及marching cube出来的mesh的质量，没记错有dual marching cube，还有处理marching cube的缺陷的论文邓。
## Non-linear sphere tracing for rendering deformed signed distance fields
- https://cs.dartmouth.edu/wjarosz/publications/seyb19nonlinear.html

- **Directly deforming implicits**
```
平移/旋转/伸缩球
inverse transformation(rays map to rays under affine transforms)











```
