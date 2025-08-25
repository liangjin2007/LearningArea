# MuJoCo文档
- https://mujoco.readthedocs.io/en/stable/overview.html
- [1.Introduction](#1Introduction)
- [2.KeyFeatures](#2KeyFeatures)
- [3.ModelInstances](#3ModelInstances)

## 1.Introduction
## 2.KeyFeatures
## 3.ModelInstances 
- 如何得到mjModel
```
(text editor) → MJCF/URDF file → (MuJoCo parser → mjSpec → compiler) → mjModel
(user code) → mjSpec → (MuJoCo compiler) → mjModel
MJB file → (model loader) → mjModel
```
- MJCF例子1
```
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
```

- MJCF例子2
```
<mujoco model="example">
  <default>
    <geom rgba=".8 .6 .4 1"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".6 .8 1" width="256" height="256"/>
  </asset>

  <worldbody>
    <light pos="0 1 1" dir="0 -1 -1" diffuse="1 1 1"/>
    <body pos="0 0 1">
      <joint type="ball"/>
      <geom type="capsule" size="0.06" fromto="0 0 0  0 0 -.4"/>
      <body pos="0 0 -0.4">
        <joint axis="0 1 0"/>
        <joint axis="1 0 0"/>
        <geom type="capsule" size="0.04" fromto="0 0 0  .3 0 0"/>
        <body pos=".3 0 0">
          <joint axis="0 1 0"/>
          <joint axis="0 0 1"/>
          <geom pos=".1 0 0" size="0.1 0.08 0.02" type="ellipsoid"/>
          <site name="end1" pos="0.2 0 0" size="0.01"/>
        </body>
      </body>
    </body>

    <body pos="0.3 0 0.1">
      <joint type="free"/>
      <geom size="0.07 0.1" type="cylinder"/>
      <site name="end2" pos="0 0 0.1" size="0.01"/>
    </body>
  </worldbody>

  <tendon>
    <spatial limited="true" range="0 0.6" width="0.005">
      <site site="end1"/>
      <site site="end2"/>
    </spatial>
  </tendon>
</mujoco>
```
```
思考以上为啥是7自由度的一个模拟系统
joint的类型有ball, hinge, 和 free
free joint的形状为cube
ball joint的形状为球 https://mujoco.readthedocs.io/en/stable/_static/example.mp4
hinge joint的形状为小的椭球
joint和geom都有可视化，两者貌似并不是谁代表谁的关系。
tendon用来实现弹簧效果
```

### 

