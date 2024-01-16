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
  
