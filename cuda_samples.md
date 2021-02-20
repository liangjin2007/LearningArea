## CUDA 头文件
```
// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
```

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

## GPU Assert 
```
__global__ void testKernel(int N)
{
    int gtid = blockIdx.x*blockDim.x + threadIdx.x ;
    assert(gtid < N) ;
}
```

## asyncAPI, simpleMultiCopy
概念： Asynchronous Data Transfers。 
- Insert events into CUDA stream calls.
- overlapping CPU and GPU execution.
- CUDA stream calls are asynchronous
- CPU can perform computations while GPU is executing(including DMA memcopies between the host and device)
- CPU can query CUDA events to determine whether GPU has completed tasks.
```
int *a = 0;
checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
checkCudaErrors(cudaFreeHost(a));
```
```
// Insert Events
// create cuda event handles
cudaEvent_t start, stop;
checkCudaErrors(cudaEventCreate(&start));
checkCudaErrors(cudaEventCreate(&stop));

StopWatchInterface *timer = NULL;
sdkCreateTimer(&timer);
sdkResetTimer(&timer);

checkCudaErrors(cudaDeviceSynchronize());
float gpu_time = 0.0f;

// asynchronously issue work to the GPU (all to stream 0)
sdkStartTimer(&timer);
cudaEventRecord(start, 0);
cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
cudaEventRecord(stop, 0);
sdkStopTimer(&timer);

// have CPU do some work while waiting for stage 1 to finish
unsigned long int counter=0;

while (cudaEventQuery(stop) == cudaErrorNotReady)
{
    counter++;
}

checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

// print the cpu and gpu times
printf("time spent executing by the GPU: %.2f\n", gpu_time);
printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);
```

```
for (int i =0; i<STREAM_COUNT; ++i)
{

    checkCudaErrors(cudaHostAlloc(&h_data_in[i], memsize,
                                  cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_data_in[i], memsize));
    checkCudaErrors(cudaMemset(d_data_in[i], 0, memsize));

    checkCudaErrors(cudaHostAlloc(&h_data_out[i], memsize,
                                  cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_data_out[i], memsize));

    checkCudaErrors(cudaStreamCreate(&stream[i]));
    checkCudaErrors(cudaEventCreate(&cycleDone[i]));

    cudaEventRecord(cycleDone[i], stream[i]);
}

cudaEventCreate(&start);
cudaEventCreate(&stop);

init();

// Kernel warmup
incKernel<<<grid, block>>>(d_data_out[0], d_data_in[0], N, inner_reps);


// Time copies and kernel
cudaEventRecord(start,0);
checkCudaErrors(cudaMemcpyAsync(d_data_in[0], h_data_in[0], memsize,
                                cudaMemcpyHostToDevice,0));
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);

float memcpy_h2d_time;
cudaEventElapsedTime(&memcpy_h2d_time, start, stop);

cudaEventRecord(start,0);
checkCudaErrors(cudaMemcpyAsync(h_data_out[0], d_data_out[0], memsize,
                                cudaMemcpyDeviceToHost, 0));
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);

float memcpy_d2h_time;
cudaEventElapsedTime(&memcpy_d2h_time, start, stop);

cudaEventRecord(start,0);
incKernel<<<grid, block,0,0>>>(d_data_out[0], d_data_in[0], N, inner_reps);
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);

float kernel_time;
cudaEventElapsedTime(&kernel_time, start, stop);
```
