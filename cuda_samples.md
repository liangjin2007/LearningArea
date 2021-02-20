# CUDA_Samples.pdf Key Concepts and Associated Samples

## CUDA 头文件
```
// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
```

## Texture
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

## Asynchronous Data Transfers
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

## Atomic Intrinsics
概念： atomic intrinsics
```
    unsigned int numThreads = 256;
    unsigned int numBlocks = 64;
    unsigned int numData = 11;
    unsigned int memSize = sizeof(int) * numData;

    //allocate mem for the result on host side
    int *hOData;
    checkCudaErrors(cudaMallocHost(&hOData, memSize));

    //initialize the memory
    for (unsigned int i = 0; i < numData; i++)
        hOData[i] = 0;

    //To make the AND and XOR tests generate something other than 0...
    hOData[8] = hOData[10] = 0xff;

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // allocate device memory for result
    int *dOData;
    checkCudaErrors(cudaMalloc((void **) &dOData, memSize));
    // copy host memory to device to initialize to zero
    checkCudaErrors(cudaMemcpyAsync(dOData, hOData, memSize,
                                    cudaMemcpyHostToDevice, stream));

    // execute the kernel
    testKernel<<<numBlocks, numThreads, 0, stream>>>(dOData);

    //Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(hOData, dOData, memSize,
                                    cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
```
```
   // access thread id
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Test various atomic instructions

    // Arithmetic atomic instructions

    // Atomic addition
    atomicAdd(&g_odata[0], 10);

    // Atomic subtraction (final should be 0)
    atomicSub(&g_odata[1], 10);

    // Atomic exchange
    atomicExch(&g_odata[2], tid);

    // Atomic maximum
    atomicMax(&g_odata[3], tid);

    // Atomic minimum
    atomicMin(&g_odata[4], tid);

    // Atomic increment (modulo 17+1)
    atomicInc((unsigned int *)&g_odata[5], 17);

    // Atomic decrement
    atomicDec((unsigned int *)&g_odata[6], 137);

    // Atomic compare-and-swap
    atomicCAS(&g_odata[7], tid-1, tid);

    // Bitwise atomic instructions

    // Atomic AND
    atomicAnd(&g_odata[8], 2*tid+7);

    // Atomic OR
    atomicOr(&g_odata[9], 1 << tid);

    // Atomic XOR
    atomicXor(&g_odata[10], tid);
```

## C++ Function Overloading
## C++ Template
```
template<class T>
__global xxx(T *data,...)
```
## CUBLAS
- old manner
```
   checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads, 0, stream>>>(d_C, d_A, d_B,
                                                dimsA.x, dimsB.x);
    } else {
        MatrixMulCUDA<32> <<< grid, threads, 0, stream>>>(d_C, d_A, d_B,
                                                dimsA.x, dimsB.x);
    }

    printf("done\n");
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel
    int nIter = 300;

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16> <<<grid, threads, 0, stream>>>(d_C, d_A, d_B,
                                                    dimsA.x, dimsB.x);
        } else {
            MatrixMulCUDA<32> <<<grid, threads, 0, stream>>>(d_C, d_A, d_B,
                                                    dimsA.x, dimsB.x);
        }
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
```

- cublas library
```
#include <cublas_v2.h>

cublasHandle_t handle;
cudaEvent_t start, stop;

checkCudaErrors(cublasCreate(&handle));

//Perform warmup operation with cublas
checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));

// Allocate CUDA events that we'll use for timing
checkCudaErrors(cudaEventCreate(&start));
checkCudaErrors(cudaEventCreate(&stop));

// Record the start event
checkCudaErrors(cudaEventRecord(start, NULL));

for (int j = 0; j < nIter; j++)
{
    //note cublas is column primary!
    //need to transpose the order
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));

}

printf("done.\n");

// Record the stop event
checkCudaErrors(cudaEventRecord(stop, NULL));

// Wait for the stop event to complete
checkCudaErrors(cudaEventSynchronize(stop));

float msecTotal = 0.0f;
checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

```

## CUBLAS Library
```
cublasXgemmBatched
cublasXgemm
```
## CUBLAS-XT Library
## CUDA Driver API
```
比如inlinePTX使用汇编api, asm() . @See inlinePTX.cpu
比如使用Driver API cuModuleLoad从cubin文件读取kernel代码或者使用cuModuleLoadDataEx从ptx文件读取kernel代码。(cuModuleGetFunction)。驱动api一般以cu开头。
cuMemcpy etc.
```
