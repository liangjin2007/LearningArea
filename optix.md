# optix7course

### ex01_helloOptix
```
try{
cudaFree(0); // ??
cudaGetDeviceCount(&numDevices);
optixInit();
}catch(std::runtime_error& e){
}
```


### ex02_pipelineAndRayGen
- pipeline
```
initOptix();

createContext(); // setup a device, use cuCtxGetCurrent(&cudaContext) to get current cuda context, then use cuda context to create optixcontext, optixDeviceContextCreate

createModule();  // create a module to contain the cu program. generally there may be multiple cu files. here using a single embedded ptx string.                  optixModuleCreateFromPTX need optix context, module compile options, pipline compile options(Payload, motionBlur, pipelineLaunchParamsVariableName, traversableGraphFlags), OptixModule object created.

createRaygenPrograms(); // optixProgramGroupCreate.  指定kind, raygen.module, raygen.entryFunctionName这个是kernel函数名
createMissPrograms(); // kind, miss.module, miss.entryFunctionName
createHitgroupPrograms(); // kind, hitgroup.moduleCH, hitgroup.entryFunctionNameCH, hitgroup.moduleAH, higroup.entryFunctionNameAH, CH means closestHit, AH means any hit.

// 上面这三个函数得到一些OptimProgramGroup对象。
createPipeline(); // optixPipelineCreate得到一个OptixPipeline对象。 optixPipelineSetStackSize(pipeline, ...)

buildSBT(); // shader binding table. build raygen records, build miss records, build hit group records. Setup a OptixShaderBindingTable object's raygenRecord
            // e.g. sbt.raygenRecord = global memory pointer, this point's length = sizeof(RayGenRecord)*rayGenCount. 
            // sbt.missRecordBase
            // sbt.missRecordStrideInBytes
            // sbt.missRecordCount
            // sbt.hitgroupRecordBase
            // sbt.higgroupStrideInBytes
            // sbt.hitgroupRecordCount
            
// 每帧会调用
optixLaunch(/*! pipeline we're launching launch: */
                            pipeline,stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes,
                            &sbt,
                            /*! dimensions of the launch: */
                            launchParams.fbSize.x,
                            launchParams.fbSize.y,
                            1
                            );
```

- 阅读代码产生的问题
```
问题1：
Shader文件 devicePrograms.cu中需要写一些kernel函数用来设置SBT， 即使这些kernel函数是空的也行。
这个文件需要手工编译成embeded_ptx_code？ 怎么编译？

```

- 第一个raygen kernel/shader程序
```
  extern "C" __global__ void __raygen__renderFrame()
  {
    if (optixLaunchParams.frameID == 0 &&
        optixGetLaunchIndex().x == 0 &&
        optixGetLaunchIndex().y == 0) {
      // we could of course also have used optixGetLaunchDims to query
      // the launch size, but accessing the optixLaunchParams here
      // makes sure they're not getting optimized away (because
      // otherwise they'd not get used)
      printf("############################################\n");
      printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
             optixLaunchParams.fbSize.x,
             optixLaunchParams.fbSize.y);
      printf("############################################\n");
    }

    // ------------------------------------------------------------------
    // for this example, produce a simple test pattern:
    // ------------------------------------------------------------------

    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int r = (ix % 256);
    const int g = (iy % 256);
    const int b = ((ix+iy) % 256);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix+iy*optixLaunchParams.fbSize.x;
    optixLaunchParams.colorBuffer[fbIndex] = rgba;
  }
```

- 写出图片
```
main.cpp中在render()结束后会用cuda download image。然后做写出到硬盘的操作。 
```


### ex03_testFrameInWindow
- 一、使用glfwwindow建立一个基于opengl的实时渲染框架
```
继承一个GLFWindow对象，重载一堆虚函数即可。
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>
  struct SampleWindow : public GLFWindow
  {
    SampleWindow(const std::string &title)
      : GLFWindow(title)
    {}
    
    virtual void render() override
    {
      sample.render();
    }
    
    virtual void draw() override
    {
      sample.downloadPixels(pixels.data());
      if (fbTexture == 0)
        glGenTextures(1, &fbTexture);
      
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      GLenum texFormat = GL_RGBA;
      GLenum texelType = GL_UNSIGNED_BYTE;
      glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                   texelType, pixels.data());

      glDisable(GL_LIGHTING);
      glColor3f(1, 1, 1);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      
      glDisable(GL_DEPTH_TEST);

      glViewport(0, 0, fbSize.x, fbSize.y);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

      glBegin(GL_QUADS);
      {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
      
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
      }
      glEnd();
    }
    
    virtual void resize(const vec2i &newSize) 
    {
      fbSize = newSize;
      sample.resize(newSize);
      pixels.resize(newSize.x*newSize.y);
    }

    vec2i                 fbSize;
    GLuint                fbTexture {0};
    SampleRenderer        sample;
    std::vector<uint32_t> pixels;
  };
```
