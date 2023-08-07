# 1小时教程 https://yuanming.taichi.graphics/publication/2020-taichi-tutorial/taichi-tutorial.pdf


```
安装，CLI，架构，优点，使用

pip install taichi -U

例子 ti gallery


///-----------------使用-------------------
1. ti.init(arch=ti.cuda)

2. 数据类型 ti.i8/i16/i32/i64/u8/u16/u32/u64/f32/f64

  type casts
    implicit casts : 稍微跟原生的python有点不一样
    ti.casts: variable = ti.casts(variable, type)

  
  Compound type
    vector, matrix, struct
    vec3f = ti.types.vector(3, ti.f32)
    mat2f = ti.types.matrix(2, 2, ti.f32)
    ray = ti.types.struct(ro = vec3f, rd = vec3f, l = ti.i32)


  Predefined keywords for compound type
    ti.Vector/ti.Matrix/ti.Struct



3. 面向数据的编程语言 ti.field(dtype=ti.f32, shape=(400, 200)) 
  a global N-d array of elements
  pixels = ti.field(dtype = float, shape=(400, 200))
  vf = ti.Vector.field(3, ti.f32, shape = 4)   # 4 x 1 的向量的field， 每个元素是3 x 1的向量
  scalar = ti.field(dtype=ti.f32, shape=())    # 标量
  scalar[None] = 5.

4. kernel @ti.kernel  
compiled, statically-typed, lexically-scoped, parallel and differentiable
@ti.kernel
def calc(i : ti.i32) -> ti.i32:
  s = 0
  return s

5. function @ti.func
不需要type-hinted
force-inlined
只允许一个return
不允许递归

6. math ti.cos(x)
支持链式比较

7.矩阵和线性代数 ti.Matrix, ti.Vector
ti.svd
ti.polar_decompose

8. parallel for-loops
  range-for loops
  struct-for loops

9. atomic operations
  += etc

10. taichi scope vs python scope

11. phases of a taichi program
  1 Initialization: ti.init(...)
  2 Field allocation: ti.field, ti.Vector.field, ti.Matrix.field
  3 Computation (launch kernels, access fields in Python-scope)
  4 Optional: restart the Taichi system (clear memory, destroy all variables and
  kernels): ti.reset()

12.  objective data-oriented programming
@ti.data_oriented
@ti.func
@ti.kernel
@staticmethod

13. meta programming
  1 templates kernel by using parameter with type ti.template()
    @ti.kernel
    def copy(x: ti.template(), y: ti.template(), c: ti.f32):
      for i in x:
        y[i] = x[i] + c

  2 template kernel instantiation
  
  3 Field-size reflection
    ti.init()
    field = ti.field(dtype=ti.f32, shape=(4, 8, 16, 32, 64))
    @ti.kernel
    def print_shape(x: ti.template()):
      ti.static_print(x.shape)
      for i in ti.static(range(len(x.shape))):
        print(x.shape[i])
        print_shape(field)

  4 Compile-time branching
  
  5 Forced loop-unrolling

  6 Variable aliasing


14 Differentiable Programming

15 Debugging

16 Visualization

```


# https://taichi.graphics/
# https://docs.taichi-lang.org/docs/tutorial
# https://docs.taichi-lang.org/tgc01/
# https://space.bilibili.com/1779922645/channel/seriesdetail?sid=337716&ctype=0
# https://www.zhihu.com/question/429149717
