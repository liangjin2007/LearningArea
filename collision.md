# Collision Detection

- collision-detection cuda
  - broad-phase collision detection https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
  - lcp algorithm for collision detection https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-33-lcp-algorithms-collision-detection-using-cuda

## BVH

```
spheres
aabb
obb
k-dops
spherical shells
```

- [2007]Early Split Clipping for Bounding Volume Hierarchies
```
这篇paper把BVH的算法流程画得很清楚
SAH被用来选择分离面
```

## Octree
- 周昆教授有一篇论文是 Data Parallel Octrees for Surface Reconstruction 
- Octree的改进版LooseOctree

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
- [2010][Min Tang]Fast Continuous Collision Detection using Deforming Non-Penetration Filters. http://gamma.cs.unc.edu/DNF/

- [2014]Defending Continuous Collision Detection against Errors
```
error tolerance
failure-proof




```

- [2015]TightCCD


