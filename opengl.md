https://learnopengl.com/book/book_pdf.pdf

## GL Buffer
glGenBuffers(1, &buffer);
glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer);
glBufferData(GL_PIXEL_PACK_BUFFER, size, NULL, GL_DYNAMIC_COPY); // frequently modification
glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);


## 各种各样的拷贝

- glCopyPixels
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
