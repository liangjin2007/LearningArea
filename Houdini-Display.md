# 显示/渲染相关

## [VEX](https://www.sidefx.com/docs/houdini/vex/index.html)
### 介绍
```
1. VEX不是scripting的可选项，但是是一个用于写shader和custom节点的更高效，小巧的一般目的语言。
2. 用途
  Rendering : Mantra使用VEX来做所有shading计算，包括light, surface, displacement and fog shaders
  Compositing: VEX Generator COP 和 VEX Filter COP。 比Pixel Expression COP要快1000倍。
  Particles： POP VOP
  Modeling: VEX SOP
  CHOPs: VEX CHOP
  Fur
```
### VEX 语言参考
C-style跟C超级像。

Context

printf

statements
```
数据类型： int, float, vector, vector2, vector4, array, struct, matrix2, matrix3, matrix, string, dict, bsdf
  Structs
  Struct functions

类型转化：结果为前者的类型
  float * int
  int * float


pragmas


{}

函数必须前向申明
无递归函数
但递归算法可以用shader calls实现
函数参数传递的是引用
可以直接试用全局变量


Main function
surface noise_surf(vector clr = {}; float frequency = 1; export vector nml = {0, 0, 0})
{
  xxx
  nml = xxx;
}


if(condition) statement [else statement2]

return
break
continue

循环
  do loop
    do statement [while(condition)]
  for(init; condition; change) statement
  foreach (value; array) statement
  foreach(index; value; array) statement
  while(condition) statement

Operators
  dot operator: 向量元素.x, 矩阵元素.xx, .zz， 调配， v.zyx等价于(v.z, v.y, v.x)

比较
  应该跟c语言一致
  
```

## GLSL Shader 
https://www.sidefx.com/docs/houdini/shade/glsl.html


## HOM
```
hou.displaySetType
  SceneObject, SelectedObject, GhostObject, DisplayModel, CurrentModel, TemplateModel
hou.glShadingType
  Flat, FlatWire, Smooth, SmoothWire, MatCapWire, MatCap, HiddenLineInvisable, HiddenLineGhost, WireGhost, Wire, ShadedBoundingBox, WireBoundingBox

```

