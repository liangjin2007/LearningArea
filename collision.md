# 物理动力学模型 
[1992]Dressing Animated Synthetic Actors with Complex Deformable Clothes
```
dissipative function消散函数

E strain ： 是变形量的度量， a measure of the amount of deformation




Lagrangian strain tensor


```
# Collision Detection
- 关键词
```
interfering objects
penetration depth: 为了算collision response
interpenetratio

```
- bvh
```
BVH: construct, refitting, and rebuiding

```

- collision-detection cuda
  - broad-phase collision detection https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
  - lcp algorithm for collision detection https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-33-lcp-algorithms-collision-detection-using-cuda
```

sort and sweep method : ok

spatial subdivision: 


```

# Collision Response

```
potential field method : 
```


[1993]Issues in Computing Contact Forces for Nonpenetrating Rigid Bodies


# Continuous Collision Detection(CCD)
#### Inclusion-Based Root-Finding Method
```
[1993 Snyder etc ,2002 Redon etc], Inclusion-Based Root-Finding： 称为保守Conservative way。不会漏检（no false negative）。
```
```
区间算数： Interval arithmetic
```


#### Numerical Root-Finding
```
三次多项式的精确求解。 
[1997]年的方法最受欢迎。
三次方程用来找到共面，然后验证是否重叠来决定是否发生了碰撞。


```




