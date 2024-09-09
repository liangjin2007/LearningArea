# PBR 

About: Physics based realtime rendering. 

Reference: github PBR source code https://github.com/Nadrin/PBR/tree/master

参考的PPT：https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf

## Lambert模型
```
Lambert模型在计算机图形学中通常用于模拟物体表面的光照效果。在早期的Lambert模型中，一个物体的最终颜色（或者说是像素的颜色）通常由以下几个分量组成：
  环境光（Ambient）：这是场景中均匀照亮物体的光。它不考虑光源的位置和方向，为物体提供基础亮度，使得物体不会完全黑暗。
  漫反射（Diffuse）：这部分模拟了光线击中粗糙表面后，光线在各个方向均匀反射的现象。漫反射的强度取决于光线与表面法线的夹角，当光线垂直于表面时，漫反射最强。
  镜面反射（Specular）：这部分模拟了光线在光滑表面上的直接反射。镜面反射通常表现为亮点或高光，其强度取决于观察角度和表面的光滑程度。
除了上述分量，还有一些其他分量可能会被包括在更复杂的Lambert模型或其扩展模型中：
  高光（Specular Highlight）：虽然镜面反射中已经包含了高光的概念，但在一些模型中，高光可能会被单独处理，以更精细地模拟光线在非常光滑表面上的反射。
  发射（Emission）：某些物体可以自己发光，如光源或屏幕上的发光物体。发射分量直接为物体添加颜色，不考虑任何外部光照。
  透明度（Transparency）/折射（Refraction）：如果物体是透明的，那么光线会穿过物体并可能发生折射。这通常需要额外的计算来模拟。
  阴影（Shadows）：虽然不是颜色分量的一部分，但阴影会影响物体的最终外观。在考虑最终颜色时，通常需要考虑物体是否被其他物体遮挡，从而接收不到某些光线。
  反射（Reflection）：某些高级模型可能会考虑物体表面的反射效果，即物体反射其周围环境的能力。
需要注意的是，Lambert模型本身主要考虑的是漫反射分量，而其他分量通常是在此基础上扩展或结合其他光照模型来实现的。随着计算机图形学的发展，这些模型已经变得更加复杂和真实，能够模拟更多细节的光照效果。
```

## 预计算lightmap
```
// VS
void main() {
    // 计算世界空间中的位置和法线
    vec4 worldPosition = modelMatrix * vec4(inPosition, 1.0);
    vec3 worldNormal = normalize(mat3(modelMatrix) * inNormal);
    // 计算光照强度（简单的Lambert漫反射）
    float diff = dot(worldNormal, -lightDirection);
    // 输出光照强度作为纹理坐标，用于后续烘焙到lightmap
    // 这里简化处理，直接使用光照强度作为R通道，其他通道为0
    outTexCoord = vec2(diff, 0.0);
    // 进行实际的顶点变换
    gl_Position = projectionMatrix * viewMatrix * worldPosition;
}

// FS
// 输入：光照强度
in vec2 inTexCoord;
// 输出：lightmap颜色
layout(location = 0) out vec4 outColor;
void main() {
    // 简化处理，直接使用光照强度作为颜色输出
    outColor = vec4(inTexCoord.x, 0.0, 0.0, 1.0); // 只使用R通道，其他通道为0或1
}
```

## UE4 PBR渲染
```

```


## 源代码解读
```
三个program用来调用drawcall的时候使用。skybox program, pbr program, tonemap program

每帧的render函数做如下的事情:
  关掉depth test先渲染天空盒，
  开启depth test渲染pbr
  全屏triangle的tone mapping/post processing。

天空盒
  skybox_fs.glsl
  skybox_vs.glsl
  layout(location=0) in vec3 localPosition;
  layout(binding=0) uniform samplerCube envTexture;
  vec3 envVector = normalize(localPosition);
	color = textureLod(envTexture, envVector, 0);


Tone mapping：
    vs: 画了一个能覆盖屏幕的三角形。
    fs:
      const float gamma     = 2.2;
      const float exposure  = 1.0;
      const float pureWhite = 1.0;
      vec3 color = texture(sceneColor, screenPosition).rgb * exposure;
      vec3 luminance = dot(color, vec3(0.2126, 0.7152, 0.0722))/*luminance*/ ;
      vec3 mappedLuminance = (luminance*(1+luminance/(pureWhite*pureWhite)))/(1+luminance);
      vec3 mappedColor = (mappedLuminance / luminance) * color;
      	// Gamma correction.
	    outColor = vec4(pow(mappedColor, vec3(1.0/gamma)), 1.0);   // color^(1/gamma)



PBR material：


setup阶段按顺序有几个compute shader, 
1.equirect2cube_cs.glsl

  environment.hdr是一个球面的equirect map表示，是一个2D texture。 代码中在Setup阶段使用compute shader将其转化为cube map。 i.e. shaders/glsl/equirect2cube_cs.glsl。
  在计算机图形学中，equirectangular map（通常简称为equirect map，又称圆柱投影地图）是一种将三维球面映射到二维平面的方法。这种映射方式保持了两极之间的角度关系，使得经线和纬线在二维平面上呈现为等距的直线。
  具体来说，equirectangular map将球面按照经度和纬度划分成小格子，然后将这些格子展开成一个矩形。在这种映射中，经线（纵向线条）和纬线（横向线条）是均匀分布的，因此它非常适合表示地球表面的地图。然而，这种映射方式在两极附近会出现较大的畸变，因为靠近极点的区域在展开时会被拉得非常长。
  在计算机图形学和虚拟现实领域，equirectangular map常用于全景图像的表示。全景图像通常捕捉了360度水平视角和180度垂直视角的图像，可以用来创建环绕观者的虚拟环境。由于equirectangular map保持了角度关系，因此它可以用来创建无缝的全景视图，尽管在两极附近会有畸变，但整体上它是一种简单且易于实现的映射方法。

2.spmap_cs.glsl
3.irmap_cs.glsl
4.spbrdf_cs.glsl




```
## UE4 
- 着色模型Shading Model
  - Diffuse BRDF 
    - Lambertian Diffuse Model $f(\vec{l}, \vec{v}) = \frac {\vec{c}_{diff}}{\pi}$
    - Microfacet Specular BRDF $f(\vec{l}, \vec{v}) = D\left(\vec{h}\right) F\left(\vec{v}, \vec{h}\right) G\left(\vec{l}, \vec{v}, \vec{h}\right)$ /
      $\vec{n}$
      
```

