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

二、main函数中执行窗口的run函数
```
      SampleWindow *window = new SampleWindow("Optix 7 Course Example");
      window->run();
```

### ex04_firstTriangleMesh
- 比之前多了个OptixTraversableHandle buildAccel(const TriangleMesh& model)。 这个函数buildAccel放在createPipeline之前， 它返回的traverableHandle放在LaunchParam里面传给shader中的一个全局变量。
```
 OptixTraversableHandle SampleRenderer::buildAccel(const TriangleMesh &model)
  {
    // upload the model to the device: the builder
    vertexBuffer.alloc_and_upload(model.vertex);
    indexBuffer.alloc_and_upload(model.index);
    
    OptixTraversableHandle asHandle { 0 };
    
    // ==================================================================
    // triangle inputs
    // ==================================================================
    OptixBuildInput triangleInput = {};
    triangleInput.type
      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_vertices = vertexBuffer.d_pointer();
    CUdeviceptr d_indices  = indexBuffer.d_pointer();
      
    triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(vec3f);
    triangleInput.triangleArray.numVertices         = (int)model.vertex.size();
    triangleInput.triangleArray.vertexBuffers       = &d_vertices;
    
    triangleInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes  = sizeof(vec3i);
    triangleInput.triangleArray.numIndexTriplets    = (int)model.index.size();
    triangleInput.triangleArray.indexBuffer         = d_indices;
    
    uint32_t triangleInputFlags[1] = { 0 };
    
    // in this example we have one SBT entry, and no per-primitive
    // materials:
    triangleInput.triangleArray.flags               = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords               = 1;
    triangleInput.triangleArray.sbtIndexOffsetBuffer        = 0; 
    triangleInput.triangleArray.sbtIndexOffsetSizeInBytes   = 0; 
    triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0; 
      
    // ==================================================================
    // BLAS setup
    // ==================================================================
    
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE
      | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
      ;
    accelOptions.motionOptions.numKeys  = 1;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixContext,
                 &accelOptions,
                 &triangleInput,
                 1,  // num_build_inputs
                 &blasBufferSizes
                 ));
    
    // ==================================================================
    // prepare compaction
    // ==================================================================
    
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));
    
    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();
    
    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    
    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    
    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
      
    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* stream */0,
                                &accelOptions,
                                &triangleInput,
                                1,  
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,
                                
                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,
                                
                                &asHandle,
                                
                                &emitDesc,1
                                ));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize,1);
    
    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
                                  /*stream:*/0,
                                  asHandle,
                                  asBuffer.d_pointer(),
                                  asBuffer.sizeInBytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
  }
  
```

- __raygen__renderFrame()使用了optixLaunchParams的traversable, camera.position, camera.direction, frame.size, frame.colorBuffer调用了optixTrace去取得u0, u1??
```
extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto &camera = optixLaunchParams.camera;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    vec3f pixelColorPRD = vec3f(0.f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );

    // normalized screen plane position, in [0,1]^2
    const vec2f screen(vec2f(ix+.5f,iy+.5f)
                       / vec2f(optixLaunchParams.frame.size));
    
    // generate ray direction
    vec3f rayDir = normalize(camera.direction
                             + (screen.x - 0.5f) * camera.horizontal
                             + (screen.y - 0.5f) * camera.vertical);

    optixTrace(optixLaunchParams.traversable,
               camera.position,
               rayDir,
               0.f,    // tmin
               1e20f,  // tmax
               0.0f,   // rayTime
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,             // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               SURFACE_RAY_TYPE,             // missSBTIndex 
               u0, u1 );

    const int r = int(255.99f*pixelColorPRD.x);
    const int g = int(255.99f*pixelColorPRD.y);
    const int b = int(255.99f*pixelColorPRD.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
  }
```

- 消化一下其他几个地方的代码, optixGetPayload_0取得的是optixTrace的结果？？
```
 template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  
  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__radiance()
  {
    const int   primID = optixGetPrimitiveIndex();
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    prd = gdt::randomColor(primID);
  }
  
  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */ }


  
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    // set to constant white as background color
    prd = vec3f(1.f);
  }
```


### ex05_firstSBTData
- Host code
```
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    TriangleMeshSBTData data;
  };
  
  void buildSBT(){
                ...
                // ------------------------------------------------------------------
                // build hitgroup records
                // ------------------------------------------------------------------
                int numObjects = 1;
                std::vector<HitgroupRecord> hitgroupRecords;
                for (int i=0;i<numObjects;i++) {
                  // we only have a single object type so far
                  int objectType = 0;
                  HitgroupRecord rec;
                  OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType],&rec));
                  rec.data.vertex = (vec3f*)vertexBuffer.d_pointer();
                  rec.data.index  = (vec3i*)indexBuffer.d_pointer();
                  rec.data.color  = model.color;
                  hitgroupRecords.push_back(rec);
                }
                hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
                sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
                sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
                sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
              
  }
```

- shader code modification
```
  extern "C" __global__ void __closesthit__radiance()
  {
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // compute normal:
    const int   primID = optixGetPrimitiveIndex();
    const vec3i index  = sbtData.index[primID];
    const vec3f &A     = sbtData.vertex[index.x];
    const vec3f &B     = sbtData.vertex[index.y];
    const vec3f &C     = sbtData.vertex[index.z];
    const vec3f Ng     = normalize(cross(B-A,C-A));

    const vec3f rayDir = optixGetWorldRayDirection();
    const float cosDN  = 0.2f + .8f*fabsf(dot(rayDir,Ng));
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    prd = cosDN * sbtData.color;
  }
```


### ex06_multipleObjects

```
    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = (int)meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID=0;meshID<numObjects;meshID++) {
      HitgroupRecord rec;
      // all meshes use the same code, so all same hit group
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0],&rec));
      rec.data.color  = meshes[meshID].color;
      rec.data.vertex = (vec3f*)vertexBuffer[meshID].d_pointer();
      rec.data.index  = (vec3i*)indexBuffer[meshID].d_pointer();
      hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
```

- buildAccel
```
OptixTraversableHandle SampleRenderer::buildAccel()
  {
    // meshes.resize(1);

    vertexBuffer.resize(meshes.size());
    indexBuffer.resize(meshes.size());
    
    OptixTraversableHandle asHandle { 0 };
    
    // ==================================================================
    // triangle inputs
    // ==================================================================
	std::vector<OptixBuildInput> triangleInput(meshes.size());
    std::vector<CUdeviceptr> d_vertices(meshes.size());
	std::vector<CUdeviceptr> d_indices(meshes.size());
	std::vector<uint32_t> triangleInputFlags(meshes.size());

    for (int meshID=0;meshID<meshes.size();meshID++) {
    // upload the model to the device: the builder
    TriangleMesh &model = meshes[meshID];
    vertexBuffer[meshID].alloc_and_upload(model.vertex);
    indexBuffer[meshID].alloc_and_upload(model.index);

    triangleInput[meshID] = {};
    triangleInput[meshID].type
      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
    d_indices[meshID]  = indexBuffer[meshID].d_pointer();
      
    triangleInput[meshID].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
    triangleInput[meshID].triangleArray.numVertices         = (int)model.vertex.size();
    triangleInput[meshID].triangleArray.vertexBuffers       = &d_vertices[meshID];
    
    triangleInput[meshID].triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput[meshID].triangleArray.indexStrideInBytes  = sizeof(vec3i);
    triangleInput[meshID].triangleArray.numIndexTriplets    = (int)model.index.size();
    triangleInput[meshID].triangleArray.indexBuffer         = d_indices[meshID];
    
    triangleInputFlags[meshID] = 0 ;
    
    // in this example we have one SBT entry, and no per-primitive
    // materials:
    triangleInput[meshID].triangleArray.flags               = &triangleInputFlags[meshID];
    triangleInput[meshID].triangleArray.numSbtRecords               = 1;
    triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer        = 0; 
    triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes   = 0; 
    triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0; 
    }
    // ==================================================================
    // BLAS setup
    // ==================================================================
    
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE
      | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
      ;
    accelOptions.motionOptions.numKeys  = 1;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixContext,
                 &accelOptions,
                 triangleInput.data(),
                 (int)meshes.size(),  // num_build_inputs
                 &blasBufferSizes
                 ));
    
    // ==================================================================
    // prepare compaction
    // ==================================================================
    
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));
    
    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();
    
    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    
    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    
    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
      
    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* stream */0,
                                &accelOptions,
                                triangleInput.data(),
                                (int)meshes.size(),
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,
                                
                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,
                                
                                &asHandle,
                                
                                &emitDesc,1
                                ));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize,1);
    
    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
                                  /*stream:*/0,
                                  asHandle,
                                  asBuffer.d_pointer(),
                                  asBuffer.sizeInBytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();
    
    return asHandle;
  }
```


### ex07_firstRealModel
这个没啥特别的

### ex08_addingTextures
- host code
```
  struct TriangleMeshSBTData {
    vec3f  color;
    vec3f *vertex;
    vec3f *normal;
    vec2f *texcoord;
    vec3i *index;
    bool                hasTexture;
    cudaTextureObject_t texture;
  };
SampleRenderer
{
...
    std::vector<CUDABuffer> normalBuffer;
    std::vector<CUDABuffer> texcoordBuffer;
    std::vector<cudaArray_t>         textureArrays;
    std::vector<cudaTextureObject_t> textureObjects;
...    
};
  struct Texture {
    ~Texture()
    { if (pixel) delete[] pixel; }
    
    uint32_t *pixel      { nullptr };
    vec2i     resolution { -1 };
  };
  
```

- shader code
```
    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = (int)model->meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID=0;meshID<numObjects;meshID++) {
      auto mesh = model->meshes[meshID];
      
      HitgroupRecord rec;
      // all meshes use the same code, so all same hit group
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0],&rec));
      rec.data.color   = mesh->diffuse;
      if (mesh->diffuseTextureID >= 0) {
        rec.data.hasTexture = true;
        rec.data.texture    = textureObjects[mesh->diffuseTextureID];
      } else {
        rec.data.hasTexture = false;
      }
      rec.data.index    = (vec3i*)indexBuffer[meshID].d_pointer();
      rec.data.vertex   = (vec3f*)vertexBuffer[meshID].d_pointer();
      rec.data.normal   = (vec3f*)normalBuffer[meshID].d_pointer();
      rec.data.texcoord = (vec2f*)texcoordBuffer[meshID].d_pointer();
      hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
```

```
extern "C" __global__ void __closesthit__radiance()
  {
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
    
    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int   primID = optixGetPrimitiveIndex();
    const vec3i index  = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    vec3f N;
    if (sbtData.normal) {
      N = (1.f-u-v) * sbtData.normal[index.x]
        +         u * sbtData.normal[index.y]
        +         v * sbtData.normal[index.z];
    } else {
      const vec3f &A     = sbtData.vertex[index.x];
      const vec3f &B     = sbtData.vertex[index.y];
      const vec3f &C     = sbtData.vertex[index.z];
      N                  = normalize(cross(B-A,C-A));
    }
    N = normalize(N);

    // ------------------------------------------------------------------
    // compute diffuse material color, including diffuse texture, if
    // available
    // ------------------------------------------------------------------
    vec3f diffuseColor = sbtData.color;
    if (sbtData.hasTexture && sbtData.texcoord) {
      const vec2f tc
        = (1.f-u-v) * sbtData.texcoord[index.x]
        +         u * sbtData.texcoord[index.y]
        +         v * sbtData.texcoord[index.z];
      
      vec4f fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
      diffuseColor *= (vec3f)fromTexture;
    }
    
    // ------------------------------------------------------------------
    // perform some simple "NdotD" shading
    // ------------------------------------------------------------------
    const vec3f rayDir = optixGetWorldRayDirection();
    const float cosDN  = 0.2f + .8f*fabsf(dot(rayDir,N));
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    prd = cosDN * diffuseColor;
  }
```


### ex09_shadowRays
```
    // -------------------------------------------------------
    // radiance rays
    // -------------------------------------------------------
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[RADIANCE_RAY_TYPE]
                                        ));
    if (sizeof_log > 1) PRINT(log);

    // -------------------------------------------------------
    // shadow rays: technically we don't need this hit group,
    // since we just use the miss shader to check if we were not
    // in shadow
    // -------------------------------------------------------
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[SHADOW_RAY_TYPE]
                                        ));
```

```
  extern "C" __global__ void __closesthit__radiance()
  {
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
    
    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int   primID = optixGetPrimitiveIndex();
    const vec3i index  = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    const vec3f &A     = sbtData.vertex[index.x];
    const vec3f &B     = sbtData.vertex[index.y];
    const vec3f &C     = sbtData.vertex[index.z];
    vec3f Ng = cross(B-A,C-A);
    vec3f Ns = (sbtData.normal)
      ? ((1.f-u-v) * sbtData.normal[index.x]
         +       u * sbtData.normal[index.y]
         +       v * sbtData.normal[index.z])
      : Ng;
    
    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    const vec3f rayDir = optixGetWorldRayDirection();
    
    if (dot(rayDir,Ng) > 0.f) Ng = -Ng;
    Ng = normalize(Ng);
    
    if (dot(Ng,Ns) < 0.f)
      Ns -= 2.f*dot(Ng,Ns)*Ng;
    Ns = normalize(Ns);

    // ------------------------------------------------------------------
    // compute diffuse material color, including diffuse texture, if
    // available
    // ------------------------------------------------------------------
    vec3f diffuseColor = sbtData.color;
    if (sbtData.hasTexture && sbtData.texcoord) {
      const vec2f tc
        = (1.f-u-v) * sbtData.texcoord[index.x]
        +         u * sbtData.texcoord[index.y]
        +         v * sbtData.texcoord[index.z];
      
      vec4f fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
      diffuseColor *= (vec3f)fromTexture;
    }
    
    // ------------------------------------------------------------------
    // compute shadow
    // ------------------------------------------------------------------
    const vec3f surfPos
      = (1.f-u-v) * sbtData.vertex[index.x]
      +         u * sbtData.vertex[index.y]
      +         v * sbtData.vertex[index.z];
    const vec3f lightPos(-907.108f, 2205.875f, -400.0267f);
    const vec3f lightDir = lightPos - surfPos;
    
    // trace shadow ray:
    vec3f lightVisibility = 0.f;
    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &lightVisibility, u0, u1 );
    optixTrace(optixLaunchParams.traversable,
               surfPos + 1e-3f * Ng,
               lightDir,
               1e-3f,      // tmin
               1.f-1e-3f,  // tmax
               0.0f,       // rayTime
               OptixVisibilityMask( 255 ),
               // For shadow rays: skip any/closest hit shaders and terminate on first
               // intersection with anything. The miss shader is used to mark if the
               // light was visible.
               OPTIX_RAY_FLAG_DISABLE_ANYHIT
               | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
               | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               SHADOW_RAY_TYPE,            // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               SHADOW_RAY_TYPE,            // missSBTIndex 
               u0, u1 );

    // ------------------------------------------------------------------
    // final shading: a bit of ambient, a bit of directional ambient,
    // and directional component based on shadowing
    // ------------------------------------------------------------------
    const float cosDN
      = 0.1f
      + .8f*fabsf(dot(rayDir,Ns));
    
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    prd = (.1f + (.2f + .8f*lightVisibility) * cosDN) * diffuseColor;
  }
  
```

