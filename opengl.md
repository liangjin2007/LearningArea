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
- what's the source of glDrawPixels?
- In vertex shader, will (MVP * vert_position).z be in canonical space[0, 1] or [-1, 1] ? No.
## Multiple Passes OpenGL Rendering
Use GL_BLEND
```


```


## APIs

- glDepthMask(GL_FALSE): is an OpenGL function call that is used to disable writing to the depth buffer.
- glBlendFunc(GL_ONE, GL_ONE): sfactor * current_rendered_color + dfactor * already_rendered_color;  here sfactor = 1 and dfactor = 1
- glBlendEquation(GL_MIN): default equation is GL_FUNC_ADD. GL_MIN, means min(source color, destination color)
- glDepthFunc(GL_LEQUAL);
- glDepthRange(0.0, 1.0);
-
- 
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
