# 1小时教程 https://yuanming.taichi.graphics/publication/2020-taichi-tutorial/taichi-tutorial.pdf

```
安装，CLI，架构，优点，使用

使用：
1. ti.init(arch=ti.cuda)
2. 数据类型 ti.i8/i16/i32/i64/u8/u16/u32/u64/f32/f64
3. 面向数据的编程语言 ti.field(dtype=float, shape=(400, 200)) 
4. kernel @ti.kernel  compiled, statically-typed, lexically-scoped, parallel and differentiable
@ti.kernel
def calc(i : ti.i32) -> ti.i32:
  s = 0
  return s

5. 


```
