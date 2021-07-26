# Collision Detection

- collision-detection cuda
  - broad-phase collision detection https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
  - lcp algorithm for collision detection https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-33-lcp-algorithms-collision-detection-using-cuda



# Collision Response




# Continuous Collision Detection(CCD)

#### Inclusion-Based Root-Finding Method
```
[1993 Snyder etc ,2002 Redon etc], Inclusion-Based Root-Finding： 称为保守Conservative way。不会漏检（no false negative）。
```
```
区间算数： Interval arithmetic. IIR. inclusion function: f IIR -> IIR 
```

```
separating axis theorem
```

```
conservative overlap test: 不要遗漏碰撞
```

#### Numerical Root-Finding
```
三次多项式的精确求解。 
[1997]年的方法最受欢迎。

三次方程用来找到共面，然后验证是否重叠来决定是否发生了碰撞。

rounding error会导致false negative(missing collisions)

```

#### 减少false positives
[2010][Min Tang]Fast Continuous Collision Detection using Deforming Non-Penetration Filters. http://gamma.cs.unc.edu/DNF/


#### [2015]TightCCD




# BVH

```
spheres
aabb
obb
k-dops
spherical shells
```

