# Introduction to physics based animation
- Initial Value Problem: 它是一个常微分方程
```
Xp(0) = X0
dXp(t)/dt = v(Xp(t), t)
```
- Another Initial Value Problem
```
Xp(0) = X0
Vp(0) = V0
dXp(t)/dt = v(Xp(t), t)
dVp(t)/dt = f(Xp(t), t)/m
```

# Contact and Friction Simulation for Computer Graphics
-- Siggraph 2021 Course

- 三大分类
```
1. constraint based method : related to constrained optimization
2. penalty based method : force-based modeling
3. impulse based method : micro-collisions
```

- 运动方程 equation of motion
```
M(t) 质量
q(t) 位置
u(t) 速度
f(t, q(t), u(t)) 力

牛顿-欧拉方程
M du/dt = f， 注意paper中u上1点表示关于时间t的导数
```


- 时间积分
```
time step h
explicit and implicit

f(q-, u-)

```
- 冲量、动量 
```
动量mv
冲量Ft = mv1 - mv2
角动量
```
-


```
库仑摩擦Coulomb friction 

```
