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


```
问题1：
Shader文件 devicePrograms.cu中需要写一些kernel函数用来设置SBT， 即使这些kernel函数是空的也行。
这个文件需要手工编译成embeded_ptx_code？ 怎么编译？

```


