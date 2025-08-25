# MuJoCo文档
- https://mujoco.readthedocs.io/en/stable/overview.html
- [1.Introduction](#1Introduction)
- [2.KeyFeatures](#2KeyFeatures)
- [3.ModelInstances](#3ModelInstances)
- [4.ModelElements](#4ModelElements)

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

## 4.ModelElements
### 4.1 Options
- mjOption影响物理模拟
- mjVisual可视化选项： 我们不用管
- mjStatistic：关于mjModel的统计信息，比如平均body质量，spatial extent
### 4.2 Assets
what? Assets不是Model elements，但是model elements可以引用它们。
- Mesh: 三角网格，obj文件/stl文件。
- Skin: 存粹可视化对象，不影响物理。没说支持的文件格式类型。
- Height Field: png文件
- Texture
- Material
### 4.3 Kinematic tree
```
MuJoCo simulates the dynamics of a collection of rigid bodies whose motion is usually constrained. The system state is represented in joint coordinates and the bodies are explicitly organized into kinematic trees。
Each body except for the top-level “world” body has a unique parent. Kinematic loops are not allowed; if loop joints are needed they should be modeled with equality constraints. Thus the backbone of a MuJoCo model is one or several kinematic trees formed by nested body definitions; an isolated floating body counts as a tree. Several other elements listed below are defined within a body and belong to that body. This is in contrast with the stand-alone elements listed later which cannot be associated with a single body.
```
- Body
```
Bodies have mass and inertial properties but do not have any geometric properties.
Instead geometric shapes (or geoms) are attached to the bodies.
Each body has two coordinate frames: the frame used to define it as well as to position other elements relative to it, and an inertial frame centered at the body’s center of mass and aligned with its principal axes of inertia.
The body inertia matrix is therefore diagonal in this frame.
At each time step MuJoCo computes the forward kinematics recursively, yielding all body positions and orientations in global Cartesian coordinates.
This provides the basis for all subsequent computations.
```
- Joint
```
Joints are defined within bodies. They create motion degrees of freedom (DOFs) between the body and its parent. In the absence of joints the body is welded to its parent. This is the opposite of gaming engines which use over-complete Cartesian coordinates, where joints remove DOFs instead of adding them. There are four types of joints: ball, slide, hinge, and a “free joint” which creates floating bodies. A single body can have multiple joints. In this way composite joints are created automatically, without having to define dummy bodies. The orientation components of ball and free joints are represented as unit quaternions, and all computations in MuJoCo respect the properties of quaternions.
```
  - Joint reference
```
The reference pose is a vector of joint positions stored in mjModel.qpos0.
It corresponds to the numeric values of the joints when the model is in its initial configuration.
In our earlier example the elbow was created in a bent configuration at 90° angle.
But MuJoCo does not know what an elbow is, and so by default it treats this joint configuration as having numeric value of 0.
We can override the default behavior and specify that the initial configuration corresponds to 90°, using the ref attribute of joint.
The reference values of all joints are assembled into the vector mjModel.qpos0.
Whenever the simulation is reset, the joint configuration mjData.qpos is set to mjModel.qpos0.
At runtime the joint position vector is interpreted relative to the reference pose.
In particular, the amount of spatial transformation applied by the joints is mjData.qpos - mjModel.qpos0.
This transformation is in addition to the parent-child translation and rotation offsets stored in the body elements of mjModel.
The ref attribute only applies to scalar joints (slide and hinge).
For ball joints, the quaternion saved in mjModel.qpos0 is always (1,0,0,0) which corresponds to the null rotation. For free joints, the global 3D position and quaternion of the floating body are saved in mjModel.qpos0.
```
  - Spring reference
```
This is the pose in which all joint and tendon springs achieve their resting length.
Spring forces are generated when the joint configuration deviates from the spring reference pose, and are linear in the amount of deviation.
The spring reference pose is saved in mjModel.qpos_spring.
For slide and hinge joints, the spring reference is specified with the attribute springref.
For ball and free joints, the spring reference corresponds to the initial model configuration.
```
- DOF
```
Degrees of freedom are closely related to joints, but are not in one-to-one correspondence because ball and free joints have multiple DOFs.
Think of joints as specifying positional information, and of DOFs as specifying velocity and force information.
More formally, the joint positions are coordinates over the configuration manifold of the system, while the joint velocities are coordinates over the tangent space to this manifold at the current position.
DOFs have velocity-related properties such as friction loss, damping, armature inertia. All generalized forces acting on the system are expressed in the space of DOFs.
In contrast, joints have position-related properties such as limits and spring stiffness.
DOFs are not specified directly by the user. Instead they are created by the compiler given the joints.
```
