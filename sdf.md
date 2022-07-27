# SDF基础

## sphere traced ray marching 
- https://github.com/sungiant/sdf 里面有图，关于sphere traced ray marching on sdf。比较好理解
- https://www.cl.cam.ac.uk/teaching/1819/FGraphics/1.%20Ray%20Marching%20and%20Signed%20Distance%20Fields.pdf
```
讲了SDF相关的一些东西，比如：


Raymarching signed distance fields
vec3 raymarch(vec3 pos, vec3 raydir) {
 int step = 0;
 float d = getSdf(pos);
 while (abs(d) > 0.001 && step < 50) {
 pos = pos + raydir * d;
 d = getSdf(pos); // Return sphere(pos) or any other
 step++;
 }

 return
 (step < 50) ? illuminate(pos, rayorig) : background;
}



交并差


Combining SDFs


Blending SDFs



Transforming SDF geometry by inverse transform
To rotate, translate or scale an SDF model, apply the inverse transform to the input point within your distance function.
float sphere(vec3 pt, float radius) {
 return length(pt) - radius;
}
float f(vec3 pt) {
 return sphere(pt - vec3(0, 3, 0));
}
This renders a sphere centered at (0, 3, 0).
More prosaically, assemble your local-to-world transform as usual, but apply its inverse to the pt within your distance function.


计算sdf的normal


sdf shadows
float shadow(vec3 pt) {
 vec3 lightDir = normalize(lightPos - pt);
 float kd = 1;
 int step = 0;
 for (float t = 0.1;
 t < length(lightPos - pt)
 && step < renderDepth && kd > 0.001; ) {
 float d = abs(getSDF(pt + t * lightDir));
 if (d < 0.001) {
 kd = 0;
 } else {
 kd = min(kd, 16 * d / t);
 }
 t += d;
 step++;
 }
 return kd;
}
里面有图，很好地理解Soft sdf shadow


Repeating SDF Geometry

```

## Sdf的交并差
```
Take the union of two SDFs by taking the min() of their functions.
● Take the intersection of two SDFs by taking the max() of their functions.
● The max() of function A and the negative of function B will return the difference of A - B.
```
## marching cube
- 前几年看过这方面的论文，论文都比较早期，paper涉及marching cube出来的mesh的质量，没记错有dual marching cube，还有处理marching cube的缺陷的论文邓。



# SDF编辑

## 
## Non-linear sphere tracing for rendering deformed signed distance fields
- https://cs.dartmouth.edu/wjarosz/publications/seyb19nonlinear.html

- **Directly deforming implicits**
```
平移/旋转/伸缩球
inverse transformation(rays map to rays under affine transforms)

[Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf
](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf)








```

# Deep SDF
- 2016 Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling

- 2019 DeepSDF DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation
  - https://www.youtube.com/watch?v=1iuLxJmQII0
  - source code : https://github.com/facebookresearch/DeepSDF
  
- 2021 A-SDF: Learning Disentangled Signed Distance Function for Articulated Shape Representation
 




