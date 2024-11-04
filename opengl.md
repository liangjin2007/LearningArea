- [1.书籍](https://learnopengl.com/book/book_pdf.pdf)
- [2.API](#2API)
- [3.MultiplePassesRendering](#3MultiplePassesRendering)
- [4.GLSLShader](#4GLSLShader)
- [5.Code](#5Code)
- [6.其他](#其他)
## 2.APIs

- Basic Scene Drawing
```
  // Do something

	glViewport(0, 0, w, h);
	glClearColor(0.6f, 0.8f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	glUseProgram(scene->program);
	glUniform1i(scene->u_lightmap, 0);
	glUniformMatrix4fv(scene->u_projection, 1, GL_FALSE, projection);
	glUniformMatrix4fv(scene->u_view, 1, GL_FALSE, view);

	glBindTexture(GL_TEXTURE_2D, scene->lightmap);

	glBindVertexArray(scene->vao);
	glDrawElements(GL_TRIANGLES, scene->indexCount, GL_UNSIGNED_SHORT, 0);
```

- glDepthMask(GL_FALSE): is an OpenGL function call that is used to disable writing to the depth buffer.
- glBlendFunc(GL_ONE, GL_ONE): sfactor * current_rendered_color + dfactor * already_rendered_color;  here sfactor = 1 and dfactor = 1
- glBlendEquation(GL_MIN): default equation is GL_FUNC_ADD. GL_MIN, means min(source color, destination color)
- glDepthFunc(GL_LEQUAL);
- glDepthRange(0.0, 1.0);
- glEnable(GL_COLOR_LOGIC_OP); : A logic operation is a bitwise operation that combines the pixel values in the framebuffer with the pixel values of the incoming fragments. It allows for various operations such as AND, OR, XOR, etc. to be performed on the pixel values.
- glLogicOp(GL_OR);
- glClearColor(1e30f, 1e30f, 1e30f, 1e30f);
- glClearDepth(1.0);
- glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
- glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_TRUE); // The function takes four boolean parameters that specify whether each channel is enabled or disabled for writing.
```

```  
GLuint vao, vbo, ibo;  
vaa(vertex attribute array)
- vao: glGenVertexArrays(1, &vao); glBindVertexArray(vao);
- vbo: glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, buffer_size, buffer_pointer, GL_STATIC_DRAW);
- ibo: glGenBuffers(1, &ibo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, buffer_size, buffer_pointer, GL_STATIC_DRAW);
- vaa: glEnableVertexAttribArray(attr_index);glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_t), ??);
```

```
texture
- texture: glGenTextures(1, &to); glBindTexture(GL_TEXTURE_2D, to);
```
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
```
- Setup texture: glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, emissive);
etc

- fbo:  glGenFramebuffers(2/*count*/, fb); glBindFramebuffer(GL_FRAMEBUFFER, fb[0]);
- rbo:  glGenRenderbuffers(1, &fbDepth); glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w[0], h[0]);
- bind rbo to fbo: glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, fbDepth);
- bind texture to frame buffer
```
frame buffer可以添加Attachment, 比如:
  添加Depth -> GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER
  添加Color -> GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ctx->hemisphere.fbTexture[i], 0);

默认framebuffer




```
```

```

- loaderShader
```
static GLuint loadShader(GLenum type, const char *source)
{
	GLuint shader = glCreateShader(type);
	if (shader == 0)
	{
		fprintf(stderr, "Could not create shader!\n");
		return 0;
	}
	glShaderSource(shader, 1, &source, NULL);
	glCompileShader(shader);
	GLint compiled;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
	if (!compiled)
	{
		fprintf(stderr, "Could not compile shader!\n");
		GLint infoLen = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
		if (infoLen)
		{
			char* infoLog = (char*)malloc(infoLen);
			glGetShaderInfoLog(shader, infoLen, NULL, infoLog);
			fprintf(stderr, "%s\n", infoLog);
			free(infoLog);
		}
		glDeleteShader(shader);
		return 0;
	}
	return shader;
}
```
- createProgram
```
static GLuint loadProgram(const char *vp, const char *fp, const char **attributes, int attributeCount)
{
	GLuint vertexShader = loadShader(GL_VERTEX_SHADER, vp);
	if (!vertexShader)
		return 0;
	GLuint fragmentShader = loadShader(GL_FRAGMENT_SHADER, fp);
	if (!fragmentShader)
	{
		glDeleteShader(vertexShader);
		return 0;
	}

	GLuint program = glCreateProgram();
	if (program == 0)
	{
		fprintf(stderr, "Could not create program!\n");
		return 0;
	}
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);

	for (int i = 0; i < attributeCount; i++)
		glBindAttribLocation(program, i, attributes[i]);

	glLinkProgram(program);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	GLint linked;
	glGetProgramiv(program, GL_LINK_STATUS, &linked);
	if (!linked)
	{
		fprintf(stderr, "Could not link program!\n");
		GLint infoLen = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLen);
		if (infoLen)
		{
			char* infoLog = (char*)malloc(sizeof(char) * infoLen);
			glGetProgramInfoLog(program, infoLen, NULL, infoLog);
			fprintf(stderr, "%s\n", infoLog);
			free(infoLog);
		}
		glDeleteProgram(program);
		return 0;
	}
	return program;
}
```
- Get Uniform Objects for later setup
```
scene->u_view = glGetUniformLocation(scene->program, "u_view");
scene->u_projection = glGetUniformLocation(scene->program, "u_projection");
scene->u_lightmap = glGetUniformLocation(scene->program, "u_lightmap");
```
- glsl shader
```

```

- Setup uniforms



## 3.MultiplePassesRendering
- 同一pixel有许多fragment, 硬件有fragment count limit，所以复杂场景会有假象
- **虽然在渲染pass 0时使用了自己新建的fbo n（fbo上绑了color texture和depth texture）进行绑定, 渲染时仍然会更新默认fbo 0的depth buffer（是否会更新color buffer？ 也会）。**
- 同一pixel可以有许多fragments
- fragment shader中 frag_out = xxx 其实对应的是每个被当前像素射线射到的所有三角形内对应点的fragment，**其实同一像素会被调用多次**。
- Early Depth Test: 比较fragment的深度 在fragment shader之前
- fragment shader后面还有per-fragment operations ： 各种Test只是过滤fragments.
  - Stencil Test: Stencil Buffer启用时比较fragment的stencil值与stencil buffer的stencil值
  - Depth Test:
  - Alpha Test
  - Blending Test: 谁跟谁blend
  - Dithering: per 
  - Output Merger
  ```
  If blending is enabled, this stage performs the blending operation. It takes the fragment color and combines it with the color already present in the color buffer, using the specified blending factors and blending equation.
  ```
- 得到深度 -(MVP * p).z
- frag_out
- 用glBlend功能 + render color buffer来获取depth range.
```
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquation(GL_MIN); 
    glClearColor(1e30f, 1e30f, 1e30f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
```


## 4.GLSLShader
- Opengl 450 core vertex shader
```
#version 450 core
layout(location=0) in vec3 position;
uniform mat4 model_mat;
uniform mat4 view_mat;
uniform mat4 proj_mat;

out VSOUT
{
  // self defined part
}vs_out; // extra output object

void main()
{
  gl_Position = xxx;
  vs_out.xxx = xxx;  // 
}
```

- Opengl 450 core fragment shader
```
#version 450 core
in VSOUT
{
  // should be the same with the description in vertex shader
}fs_in;
out vec4 frag_out;
void main()
{
  frag_out = xxx;
}
```
## 5.Code
- Lightmap(lightmass): [lightmapper](https://github.com/ands/lightmapper) 1.4k star
此应用烘培了AO效果为贴图。bake过程如下：
- IBL
- 风格化PBR
- Light Probe
- light Bloom
- particle systems
- simulating a light source.
- depth peeling
- hair: Kajiya BRDF
- [ILC](https://zhuanlan.zhihu.com/p/360967391) 
```
Indirect Lighting Cache
UE4在移动端即使在没有TOD的前提下，可选的GI方案也非常有限，静态物体的GI可以使用2D Lightmap来表达，而场景中的动态物体(Stationary,Movable，也包括诸如粒子、带骨骼动画的模型)等大多会采用基于三阶球谐(SH3)的静态漫反射来表达——UE4中表达SH3光照名字叫间接光照缓存(IndirectLightingCache，以下简称ILC)。
ILC基于逐物体生效，当一个物体覆盖比较大的空间范围的时候，ILC所能给出的结果往往会和周围使用诸如2D Lightmap，VLM等逐像素方案比有颜色/亮度跳变。

UE4生成的是Volume Lightmap
```
- Flux
- Enlighten
- HBAO
- VolumeGI
- GTAO
- SSR
- RayTracing
- Screen Space Shadow
- TOD变换 Time Of Day
- 破损，载具，单时段烘培
- SSRT
```
Screen Space Ray Tracing (SSRT) is an algorithm used in computer graphics to approximate the effects of global illumination by ray tracing, but in screen space rather than world space. This technique is often used in real-time rendering for applications like video games and simulations, where full global illumination ray tracing might be too computationally expensive.
The basic idea behind SSRT is to trace rays in the screen space of the final image, rather than in the 3D world space of the scene. This is done by using information that is already available in the screen space, such as depth and normal maps, to avoid the need for complex intersection tests with scene geometry.
Here's a simplified outline of the Screen Space Ray Tracing algorithm:
Gather Screen Space Information: Render the scene as normal, capturing depth, normal, and any other relevant information in screen space textures.
Ray Casting: For each pixel in the screen, cast rays in screen space. These rays are typically aligned with the view direction and can be offset based on the normal map to simulate roughness or reflections.
Intersection Testing: Instead of testing for intersections with 3D geometry, use the depth buffer to find the nearest intersection point along the ray. This is much faster than 3D intersection tests.
Shading: Once the intersection point is found, use the normal map and any other available information to compute the shading at that point. This can include diffuse, specular, and other lighting models.
Recursive Rays (Optional): Optionally, the algorithm can support recursive rays to simulate effects like reflections and refractions. However, because this is done in screen space, it's typically limited to a small number of bounces to maintain performance.
Composition: Combine the results of the ray tracing with the original scene color to produce the final image.
Advantages of Screen Space Ray Tracing include:
Performance: It's generally faster than world space ray tracing because it operates in a 2D space and uses pre-computed depth information.
Real-time capable: It can be used in real-time applications like games where frame rates need to be maintained.
Quality: It can provide better lighting and shadowing than traditional real-time lighting models, especially for materials like metal or glass.
Disadvantages of Screen Space Ray Tracing include:
Accuracy: It's an approximation and may not be as accurate as full 3D ray tracing, especially for complex scenes or interactions.
Limitations: It's limited to the information available in screen space, so it can't account for effects that require full 3D information, such as shadows cast by objects not visible in the screen space.
Aliasing: It can suffer from aliasing artifacts, which can be mitigated with super-sampling or other anti-aliasing techniques.
SSRT is a balance between visual quality and performance, and it's often used in conjunction with other rendering techniques to achieve the desired look in real-time graphics.
```
```
代码 https://github.com/jebbyk/SSRT-for-reshade/blob/main/ssrt.fx




```
## 6.其他

### 三角洲行动2024 https://www.bilibili.com/video/BV18F2oYAEp2/?spm_id_from=333.999.0.0&vd_source=8cfcd5a041c19f2f5777e1a8d78359f2
[UFSH2024]《三角洲行动》，诠释新一代游戏开发方式 | 王理川 天美Y1工作室技术美术组负责人/专家 
```
基于UE4.24
PC
  Lightmap
  Volume GI 0.25米精度
  SSRTAO 屏幕空间Ray Tracing
  SSR, Screen Space Shadow
  Ray Tracing
  DX12
MOBILE
  Lightmap, AO map
  Volume GI 1米精度
  Probe GI 低配动态物体
  Vulkan + OpenGL(低配)

不同TOD共享一套Lightmap
  人工光
  天光AO
PC：
  VT技术，保证高精度 Virtual Texture
MOBILE:
  大面积室外低精度AO map
不同LOD之间，2U的分布保持一致

Volume GI
  拟合存储不同TOD数据
    直接光的Bounce
    人工光
    天光AO
  PC：
    高密度
    GPU直接采样Probe数据结构
  
  MOBILE：
    低密度
    CPU上转换为3D texture

Probe GI
  沿着可行走区域，构建低密度四面体
  用于动态物体（手机低配）

SSRTAO Screen space Ray Tracing Ambient Occlusion

Screen Space Contact Shadow

Ray Tracing

画面标准
  基于标准ACES
  定制Look Modification Transform(LMT)

  PC: Local Exposure
  Mobile: 降低对比

  升级定制的ACES-LMT调色流程
  基于Davinci调色的影视流程引入

跨端资产LOD策略
  Static Mesh
    PC端加载所有的LOD列， MOBILE只加载比较低分辨率的一些比如LOD3到LOD5
  Foliage Type
    定制的，解耦，PC和MOBILE的植被是两套

Virtual Material虚拟材质


材质Shading
  PC
    Tiling 贴图、层混合材质服用
    Multi-ID
    Screen-Space 算法强调质感

  Mobile 高性能
    Lightmap, AO map
    Single-ID
    Pre-Bake拟合达到近似质感

双端SD/SP工作流
  SP材质库 & SD材质库 & UE材质库同步


Light Fixture Tools灯具
  PC
    IES & Light Function
    Bloom mask
    实时光各类属性支持
    Lens flare
    实时高光
    光体积雾
    Ray Tracing Reflection
    户外灯（响应TOD）
  Mobile
    高光只有IBL抓的自发光
    static 灯光
    有光无灯（中低配优化）
    户外灯只有mesh没有光  

  预设Preset=LightColor + Temperature=材质颜色
  ISM
  IBL反射（抓取自发光）
  Bake Lightmap

  人工光性能优化工具
    可视化预览Overdraw超量
    光源检测工具

光照渲染效果如何验收？
  人工光 Artifical Lights
  Sky & Fog
  Sky Light 天光
  Global Illumination
  Direction Light
  Environment Reflection
  Screen Space Reflection
  Post Processing

跨端UE编辑器改造
  双端数据同步互通
  编辑器内切换平台

海量破损
  破损方案
    Houdini
    Nvidia PhysX
    没有使用Chaos
    Pre-Bake动态数据，自动绑定骨骼，处理LOD用于下游
    破损资产生产管线

性能检测流程
  多维度采集数据。 NavMesh采样点，进行性能相关数据采集

JADE Asset Editor
  JADE CHECK 在DCC中装配，而不是到UE编辑器里去装配

JADE LINK
  Digital Content Creation(Maya/3dsmax) -> JadeLink -> Asset Info.JSON + FBX -> Game Engine
  Assets Info.JSON + Textures 与SD, SP, Game Engine胡同

跨区域团队美术生产管线
  NB NB NB

程序化管线工作流
  更多的生态种类： 生态生产管线
  更多的资产种类
  更高的分布密度
  PC专属配方
  道路路网： 道路Spline工具预设
  河流湖泊： 湖泊Spline工具预设，与周围生态关联

UI Map
  海量的UI图需要创建迭代和维护  

UE5加持影视级画面
  实时GI-Lumen
    

```
