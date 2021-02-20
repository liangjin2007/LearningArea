## simpleTexture
对图片进行旋转0.5弧度。
```
// Allocate device memory for result
float *dData = NULL;
checkCudaErrors(cudaMalloc((void **) &dData, size)); // 全局内存

// free device memory
checkCudaErrors(cudaFree(dData));
```

```
// Allocate array and copy image data
cudaChannelFormatDesc channelDesc =
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaArray *cuArray;
checkCudaErrors(cudaMallocArray(&cuArray,
                                &channelDesc,
                                width,
                                height));
checkCudaErrors(cudaMemcpyToArray(cuArray,
                                  0,
                                  0,
                                  hData,
                                  size,
                                  cudaMemcpyHostToDevice));
                                  
// free cuda array                         
checkCudaErrors(cudaFreeArray(cuArray));
```

```
// 传递tex给kernel作为参数
cudaTextureObject_t         tex;
cudaResourceDesc            texRes;
memset(&texRes,0,sizeof(cudaResourceDesc));

texRes.resType            = cudaResourceTypeArray;
texRes.res.array.array    = cuArray;

cudaTextureDesc             texDescr;
memset(&texDescr,0,sizeof(cudaTextureDesc));

texDescr.normalizedCoords = true;
texDescr.filterMode       = cudaFilterModeLinear;
texDescr.addressMode[0] = cudaAddressModeWrap;
texDescr.addressMode[1] = cudaAddressModeWrap;
texDescr.readMode = cudaReadModeElementType;

checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

dim3 dimBlock(8, 8, 1);
dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

// Warmup
transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle, tex);
// Check if kernel execution generated an error
getLastCudaError("Kernel execution failed");
checkCudaErrors(cudaDeviceSynchronize());

// Execute the kernel
transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle, tex);

// Check if kernel execution generated an error
getLastCudaError("Kernel execution failed");
checkCudaErrors(cudaDeviceSynchronize());
```

```
// Timer
StopWatchInterface *timer = NULL;
sdkCreateTimer(&timer);
sdkStartTimer(&timer);
... //Execute some kernel
sdkStopTimer(&timer);
printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
printf("%.2f Mpixels/sec\n",
       (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
sdkDeleteTimer(&timer);
```
