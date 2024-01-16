https://learnopengl.com/book/book_pdf.pdf

## GL buffer object, texture object, render buffer object


## GL Buffer
glGenBuffers(1, &buffer);
glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer);
glBufferData(GL_PIXEL_PACK_BUFFER, size, NULL, GL_DYNAMIC_COPY); // frequently modification
glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

```
can render buffer object interop with cuda?


No, renderbuffer objects in OpenGL cannot directly interoperate with CUDA. Renderbuffers are specific to the OpenGL rendering pipeline and are not directly accessible by CUDA for general-purpose computation.

However, it is possible to share data between OpenGL and CUDA using other mechanisms such as texture objects or buffer objects. These mechanisms allow for interoperability between OpenGL and CUDA, enabling efficient data sharing and processing.

To achieve interop between OpenGL and CUDA, you can use the following steps:

Create an OpenGL texture object or buffer object and bind it to an appropriate target.

Register the OpenGL object with CUDA using functions like cudaGraphicsGLRegisterImage or cudaGraphicsGLRegisterBuffer. This creates a CUDA graphics resource that represents the OpenGL object.

Map the CUDA graphics resource using cudaGraphicsMapResources to obtain a CUDA device pointer to the shared data.

Perform CUDA operations using the mapped device pointer.

Unmap the CUDA graphics resource using cudaGraphicsUnmapResources to release the CUDA device pointer.

Use the OpenGL object in OpenGL rendering operations.

When finished, unregister the CUDA graphics resource using cudaGraphicsUnregisterResource to release the shared resource.

By following these steps, you can achieve efficient data sharing between OpenGL and CUDA, allowing you to perform computations on shared data using CUDA and use the results for rendering in OpenGL.

Keep in mind that proper synchronization and resource management must be implemented to ensure correct data access and avoid conflicts between OpenGL and CUDA operations.
```

```
what's the source of glDrawPixels?


The source of glDrawPixels is typically the CPU memory (system RAM) rather than a specific OpenGL object like a texture or buffer.

glDrawPixels is used to directly draw pixel data stored in the CPU memory onto the currently bound framebuffer. It allows you to specify the pixel format, data type, width, height, and other parameters to describe the pixel data you want to draw. The pixel data is passed as a pointer to the start of the pixel data in CPU memory.

Here's a general overview of how glDrawPixels works:

Bind the desired framebuffer object using glBindFramebuffer(GL_FRAMEBUFFER, framebuffer).

Set the desired pixel unpacking parameters using glPixelStorei to specify the desired alignment, row length, skip pixels, and skip rows.

Call glDrawPixels with the appropriate parameters to specify the pixel format, data type, width, height, and pointer to the pixel data.

OpenGL reads the pixel data from the CPU memory and rasterizes it onto the currently bound framebuffer. The rasterization process converts the pixel data into fragments that are processed by subsequent stages of the rendering pipeline, such as fragment shading and blending.

Finally, the rendered pixels are displayed on the screen when the framebuffer is presented (e.g., by calling glfwSwapBuffers in a GLFW-based application).

It's important to note that glDrawPixels is considered a legacy function in modern OpenGL and is not commonly used in modern applications. Instead, it's more common to use textured geometry or framebuffer objects with attached textures or renderbuffers for efficient rendering in OpenGL.
```

```
copy GL_PIXEL_PACK_BUFFER to depth target?


To copy the contents of a GL_PIXEL_PACK_BUFFER to a depth target, you can use the following steps:

Bind the pixel pack buffer using glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo), where pbo is the ID of your pixel pack buffer.

Bind the depth target, such as a depth renderbuffer or depth texture, as the read framebuffer attachment using glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer) and then attaching the depth target using glFramebufferRenderbuffer or glFramebufferTexture2D.

Call glReadPixels to read the depth data from the pixel pack buffer into the depth target. Set the parameters such as the starting pixel coordinates, width, height, format, and type to match the desired depth target. For example:


glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
Unbind the pixel pack buffer using glBindBuffer(GL_PIXEL_PACK_BUFFER, 0).
Here's an example of how the code might look:


// Assume pbo is the ID of your pixel pack buffer and depthRenderbuffer is the ID of your depth renderbuffer

// Bind the pixel pack buffer
glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);

// Bind the framebuffer with the depth renderbuffer as the read framebuffer attachment
glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);

// Read the depth data from the pixel pack buffer into the depth renderbuffer
glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

// Unbind the pixel pack buffer
glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
Make sure to replace width, height, framebuffer, depthRenderbuffer, and pbo with the appropriate values specific to your application.

It's important to note that this method can be inefficient in certain scenarios, as reading from the GPU-bound pixel pack buffer can introduce a synchronization point. In general, it is recommended to use framebuffer objects and textures instead for efficient rendering and copying of depth data in modern OpenGL applications.


copy depth target to GL_PIXEL_PACK_BUFFER object?


To copy the contents of a depth target, such as a depth renderbuffer or depth texture, to a GL_PIXEL_PACK_BUFFER object, you can use the following steps:

Bind the depth target, such as a depth renderbuffer or depth texture, as the read framebuffer attachment using glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer) and then attaching the depth target using glFramebufferRenderbuffer or glFramebufferTexture2D.

Bind the pixel pack buffer using glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo), where pbo is the ID of your pixel pack buffer.

Call glReadPixels to read the depth data from the depth target into the pixel pack buffer. Set the parameters such as the starting pixel coordinates, width, height, format, and type to match the desired depth target. For example:


glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
Unbind the pixel pack buffer using glBindBuffer(GL_PIXEL_PACK_BUFFER, 0).
Here's an example of how the code might look:


// Assume pbo is the ID of your pixel pack buffer and depthRenderbuffer is the ID of your depth renderbuffer

// Bind the framebuffer with the depth renderbuffer as the read framebuffer attachment
glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);

// Bind the pixel pack buffer
glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);

// Read the depth data from the depth renderbuffer into the pixel pack buffer
glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

// Unbind the pixel pack buffer
glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
Make sure to replace width, height, framebuffer, depthRenderbuffer, and pbo with the appropriate values specific to your application.

Please note that the depth values obtained from glReadPixels will be in the range [0, 1] for normalized depth formats. If you require the depth values in a different range or format, you may need to process the data after reading it into the pixel pack buffer.

Also, keep in mind that reading depth data using glReadPixels can be slow due to the potential need to synchronize with the GPU. If possible, consider using framebuffer objects and textures instead for efficient rendering and copying of depth data in modern OpenGL applications.
```

## 各种各样的拷贝

- glCopyPixels  // Copy the pixels from the OpenGL framebuffer to a new region
```
        // Clear the color buffer
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw a rectangle
        glBegin(GL_QUADS);
        glVertex2f(-0.5f, -0.5f);
        glVertex2f(0.5f, -0.5f);
        glVertex2f(0.5f, 0.5f);
        glVertex2f(-0.5f, 0.5f);
        glEnd();

        // Copy the pixels from the OpenGL framebuffer to a new region
        glRasterPos2i(100, 100);
        glCopyPixels(0, 0, 200, 200, GL_COLOR);
```  

- glReadPixels
  
