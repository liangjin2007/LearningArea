# 参考 https://github.com/Nadrin/PBR/tree/master

## UE4 https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
参考的是Disney的Presentation

- Introduction
```
实时性能
减小复杂度
  参数少
  支持image-based lighting
直觉的接口
  没有使用物理参数（比如折射率）而是使用能简单理解的值。
视觉上线性 Perceptually Linear
容易精通
鲁棒
Expressive 能覆盖显示世界中99%的材质
Flexible 灵活的
```


- 着色模型Shading Model
  - Diffuse BRDF 
    - Lambertian Diffuse Model $f(\vec{l}, \vec{v}) = \frac {\vec{c}_{diff}}{\pi}$
    - Microfacet Specular BRDF $f(\vec{l}, \vec{v}) = D\left(\vec{h}\right) F\left(\vec{v}, \vec{h}\right) G\left(\vec{l}, \vec{v}, \vec{h}\right)$ /
      $4(\vec{n} \dot \vec{l})$
      
```

## 关键词
```
image-based lighting
spherical harmonic lighting
{4\left(\vec{n} \dot \vec{l}\right)\left(\vec{n} \dot \vec{v}\right)}
```
