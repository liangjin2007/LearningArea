## simpleTexture
对图片进行旋转0.5弧度。
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
