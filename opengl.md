https://learnopengl.com/book/book_pdf.pdf


## Questions
- can render buffer object interop with cuda? No. texture object or buffer object can. Related api cudaGraphicsGLRegisterImage, cudaGraphicsMapResources
- what is GL_PIXEL_PACK_BUFFER
```
glGenBuffers(1, &buffer);
glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer);
glBufferData(GL_PIXEL_PACK_BUFFER, size, NULL, GL_DYNAMIC_COPY); // frequently modification
glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
```
- what's the source of glDrawPixels? generally cpu buffer.
- In vertex shader, will (MVP * vert_position).z be in canonical space[0, 1] or [-1, 1] ? No.
- how to get each pixel's depth range ?
- how to render wireframe ?

- 
## Multiple Passes OpenGL Rendering
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

## APIs

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
## Opengl glsl Shader
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
## Effects
- light Bloom
- particle systems
- simulating a light source.
- depth peeling
```
The depth peeling process typically involves the following steps:

Render the scene normally, storing the depth and color values for each fragment.

Repeat the following steps for a specified number of peels:
a. Set up a depth range for the current peel, which excludes the depth range of the previously rendered peels.
b. Render the scene again, considering only the geometry within the current depth range.
c. Update the depth and color values only for the fragments that are closer to the camera than the previous peels.

Composite the color values of the visible fragments from all the peels, taking into account their depths and transparency.
```
