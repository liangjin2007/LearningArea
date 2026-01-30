# MuJoCo文档
- https://mujoco.readthedocs.io/en/stable/overview.html
- [1.Introduction](#1Introduction) 
  - [1.1KeyFeatures](#11KeyFeatures)
  - [1.2ModelInstances](#12ModelInstances)
  - [1.3Examples](#13Examples)
- [2.ModelElements](#2ModelElements)
- [3.Clarifications](#3Clarifications)
## 1.Introduction
### 1.1KeyFeatures
### 1.2ModelInstances 
- 如何得到mjModel
```
(text editor) → MJCF/URDF file → (MuJoCo parser → mjSpec → compiler) → mjModel
(user code) → mjSpec → (MuJoCo compiler) → mjModel
MJB file → (model loader) → mjModel
```
### 1.3Examples
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

## 2.ModelElements
### 2.1.Options
- mjOption影响物理模拟
- mjVisual可视化选项： 我们不用管
- mjStatistic：关于mjModel的统计信息，比如平均body质量，spatial extent
### 2.2.Assets
what? Assets不是Model elements，但是model elements可以引用它们。
- Mesh: 三角网格，obj文件/stl文件。
- Skin: 存粹可视化对象，不影响物理。没说支持的文件格式类型。
- Height Field: png文件
- Texture
- Material
### 2.3.Kinematic tree
```
MuJoCo simulates the dynamics of a collection of rigid bodies whose motion is usually constrained. The system state is represented in joint coordinates and the bodies are explicitly organized into kinematic trees。
Each body except for the top-level “world” body has a unique parent. Kinematic loops are not allowed; if loop joints are needed they should be modeled with equality constraints. Thus the backbone of a MuJoCo model is one or several kinematic trees formed by nested body definitions; an isolated floating body counts as a tree. Several other elements listed below are defined within a body and belong to that body. This is in contrast with the stand-alone elements listed later which cannot be associated with a single body.
```
### 2.3.1.Body
```
Bodies have mass and inertial properties but do not have any geometric properties.
Instead geometric shapes (or geoms) are attached to the bodies.
Each body has two coordinate frames: the frame used to define it as well as to position other elements relative to it, and an inertial frame centered at the body’s center of mass and aligned with its principal axes of inertia.
The body inertia matrix is therefore diagonal in this frame.
At each time step MuJoCo computes the forward kinematics recursively, yielding all body positions and orientations in global Cartesian coordinates.
This provides the basis for all subsequent computations.
```
### 2.3.2.Joint
```
Joints are defined within bodies. They create motion degrees of freedom (DOFs) between the body and its parent. In the absence of joints the body is welded to its parent. This is the opposite of gaming engines which use over-complete Cartesian coordinates, where joints remove DOFs instead of adding them. There are four types of joints: ball, slide, hinge, and a “free joint” which creates floating bodies. A single body can have multiple joints. In this way composite joints are created automatically, without having to define dummy bodies. The orientation components of ball and free joints are represented as unit quaternions, and all computations in MuJoCo respect the properties of quaternions.
```
### 2.3.3.Joint reference
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
### 2.3.4.Spring reference
```
This is the pose in which all joint and tendon springs achieve their resting length.
Spring forces are generated when the joint configuration deviates from the spring reference pose, and are linear in the amount of deviation.
The spring reference pose is saved in mjModel.qpos_spring.
For slide and hinge joints, the spring reference is specified with the attribute springref.
For ball and free joints, the spring reference corresponds to the initial model configuration.
```

### 2.3.5.DOF
```
Degrees of freedom are closely related to joints, but are not in one-to-one correspondence because ball and free joints have multiple DOFs.
Think of joints as specifying positional information, and of DOFs as specifying velocity and force information.
More formally, the joint positions are coordinates over the configuration manifold of the system, while the joint velocities are coordinates over the tangent space to this manifold at the current position.
DOFs have velocity-related properties such as friction loss, damping, armature inertia. All generalized forces acting on the system are expressed in the space of DOFs.
In contrast, joints have position-related properties such as limits and spring stiffness.
DOFs are not specified directly by the user. Instead they are created by the compiler given the joints.
```

### 2.3.6.Geometry
```
Geoms are 3D shapes rigidly attached to the bodies. Multiple geoms can be attached to the same body. This is particularly useful in light of the fact that MuJoCo only supports convex geom-geom collisions, and the only way to create non-convex objects is to represent them as a union of convex geoms. Apart from collision detection and subsequent computation of contact forces, geoms are used for rendering, as well as automatic inference of body masses and inertias when the latter are omitted. MuJoCo supports several primitive geometric shapes: plane, sphere, capsule, ellipsoid, cylinder, box. A geom can also be a mesh or a height field; this is done by referencing the corresponding asset. Geoms have a number of material properties that affect the simulation and visualization.
```

- comdim
```
1 for Frictionless contact.
3 for Regular frictional contact, opposing slip in the tangent plane.
```



### 2.3.7.Site
### 2.3.8.Camera
### 2.3.9.Light

### 2.4 Stand-alone

Here we describe the model elements which do not belong to an individual body, and therefore are described outside the kinematic tree.

### 2.4.1.Tendon
### 2.4.2.Actuator
- Transmission
```
Each actuator has a scalar length li(q)defined by the type of transmission and its parameters. The gradient 
∇li​ is an nV-dimensional vector of moment arms. It determines the mapping from scalar actuator force to joint force. The transmission properties are determined by the MuJoCo object to which the actuator is attached; the possible attachment object types are joint, tendon, jointinparent, slider-crank, site, and body.
```

- Stateful actuators, i.e., activation
```
Some actuators such as pneumatic and hydraulic cylinders as well as biological muscles have an internal state called “activation”. 
```

- [Force Generation](https://mujoco.readthedocs.io/en/stable/computation/index.html#force-generation)
```
Force generation
Each actuator generates a scalar force pi  which is some function
pi(ui,wi,li,l˙i)
Similarly to activation dynamics, the force generation mechanism is actuator-specific and cannot interact with the other actuators in the model. Currently the force is affine in the activation state when present, and in the control otherwise:
pi=(a wi  or a ui) + b0 + b1 li + b2l˙i

Here a is an actuator-specific gain parameter and b0, b1, b2 are actuator-specific bias parameters, stored in mjModel.actuator_gainprm and mjModel.actuator_biasprm respectively. Different settings of the gain and bias parameters can be used to model direct force control as well as position and velocity servos -in which case the control/activation has the meaning of reference position or velocity. 
```

```
joint: string, optional
This and the next four attributes determine the type of actuator transmission. All of them are optional, and exactly one of them must be specified. If this attribute is specified, the actuator acts on the given joint. For hinge and slide joints, the actuator length equals the joint position/angle times the first element of gear. For ball joints, the first three elements of gear define a 3d rotation axis in the child frame around which the actuator produces torque. The actuator length is defined as the dot-product between this gear axis and the angle-axis representation of the joint quaternion, and is in units of radian if gear is normalized (generally scaled by by the norm of gear). Note that after total rotation of more than π
, **the length will wrap to −π and vice-versa**. Therefore position servos for ball joints should generally use tighter limits which prevent this wrapping. For free joints, gear defines a 3d translation axis in the world frame followed by a 3d rotation axis in the child frame. The actuator generates force and torque relative to the specified axes. The actuator length for free joints is defined as zero (so it should not be used with position servos).
```


- Sensor
- Equality
- Flex
- Contact pair
- Contact exclude
- Custom numeric
- Custom text
- Custom tuple
- Keyframe

## 3.Clarifications

  
