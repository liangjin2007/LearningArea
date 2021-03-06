# CUDA_Samples.pdf Key Concepts and Associated Samples

# 基本概念

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
## CUDA Dynamic Parallelism

## CUDA Graphs
```
cudaStream_t streamForGraph;
cudaGraph_t graph;
std::vector<cudaGraphNode_t> nodeDependencies;
cudaGraphNode_t memcpyNode, kernelNode, memsetNode;
double result_h = 0.0;

checkCudaErrors(cudaStreamCreateWithFlags(&streamForGraph, cudaStreamNonBlocking));

cudaKernelNodeParams kernelNodeParams = {0};
cudaMemcpy3DParms memcpyParams = {0};
cudaMemsetParams memsetParams = {0};

memcpyParams.srcArray = NULL;
memcpyParams.srcPos   = make_cudaPos(0,0,0);
memcpyParams.srcPtr   = make_cudaPitchedPtr(inputVec_h, sizeof(float)*inputSize, inputSize, 1);
memcpyParams.dstArray = NULL;
memcpyParams.dstPos   = make_cudaPos(0,0,0);
memcpyParams.dstPtr   = make_cudaPitchedPtr(inputVec_d, sizeof(float)*inputSize, inputSize, 1);
memcpyParams.extent   = make_cudaExtent(sizeof(float)*inputSize, 1, 1);
memcpyParams.kind     = cudaMemcpyHostToDevice;

memsetParams.dst            = (void*)outputVec_d;
memsetParams.value          = 0;
memsetParams.pitch          = 0;
memsetParams.elementSize    = sizeof(float); // elementSize can be max 4 bytes
memsetParams.width          = numOfBlocks*2; 
memsetParams.height         = 1;

checkCudaErrors(cudaGraphCreate(&graph, 0));
checkCudaErrors(cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
checkCudaErrors(cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

nodeDependencies.push_back(memsetNode);
nodeDependencies.push_back(memcpyNode);

void *kernelArgs[4] = {(void*)&inputVec_d, (void*)&outputVec_d, &inputSize, &numOfBlocks};

kernelNodeParams.func = (void*)reduce;
kernelNodeParams.gridDim  = dim3(numOfBlocks, 1, 1);
kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
kernelNodeParams.sharedMemBytes = 0;
kernelNodeParams.kernelParams = (void **)kernelArgs;
kernelNodeParams.extra = NULL;

checkCudaErrors(cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams));

nodeDependencies.clear();
nodeDependencies.push_back(kernelNode);

memset(&memsetParams, 0, sizeof(memsetParams));
memsetParams.dst            = result_d;
memsetParams.value          = 0;
memsetParams.elementSize    = sizeof(float);
memsetParams.width          = 2;
memsetParams.height         = 1;
checkCudaErrors(cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

nodeDependencies.push_back(memsetNode);

memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
kernelNodeParams.func = (void*)reduceFinal;
kernelNodeParams.gridDim  = dim3(1, 1, 1);
kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
kernelNodeParams.sharedMemBytes = 0;
void *kernelArgs2[3] =  {(void*)&outputVec_d, (void*)&result_d, &numOfBlocks};
kernelNodeParams.kernelParams = kernelArgs2;
kernelNodeParams.extra = NULL;

checkCudaErrors(cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams));
nodeDependencies.clear();
nodeDependencies.push_back(kernelNode);

memset(&memcpyParams, 0, sizeof(memcpyParams));

memcpyParams.srcArray = NULL;
memcpyParams.srcPos   = make_cudaPos(0,0,0);
memcpyParams.srcPtr   = make_cudaPitchedPtr(result_d, sizeof(double), 1, 1);
memcpyParams.dstArray = NULL;
memcpyParams.dstPos   = make_cudaPos(0,0,0);
memcpyParams.dstPtr   = make_cudaPitchedPtr(&result_h, sizeof(double), 1, 1);
memcpyParams.extent   = make_cudaExtent(sizeof(double), 1, 1);
memcpyParams.kind     = cudaMemcpyDeviceToHost;
checkCudaErrors(cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams));
nodeDependencies.clear();
nodeDependencies.push_back(memcpyNode);

cudaGraphNode_t *nodes = NULL;
size_t numNodes = 0;
checkCudaErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
printf("\nNum of nodes in the graph created manually = %zu\n", numNodes);

cudaGraphExec_t graphExec;
checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

cudaGraph_t clonedGraph;
cudaGraphExec_t clonedGraphExec;
checkCudaErrors(cudaGraphClone(&clonedGraph, graph));
checkCudaErrors(cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

for (int i=0; i < GRAPH_LAUNCH_ITERATIONS; i++)
{
   checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
   checkCudaErrors(cudaStreamSynchronize(streamForGraph));
   printf("[cudaGraphsManual] final reduced sum = %lf\n", result_h);
   result_h = 0.0;
}

printf("Cloned Graph Output.. \n");
for (int i=0; i < GRAPH_LAUNCH_ITERATIONS; i++)
{
   checkCudaErrors(cudaGraphLaunch(clonedGraphExec, streamForGraph));
   checkCudaErrors(cudaStreamSynchronize(streamForGraph));
   printf("[cudaGraphsManual] final reduced sum = %lf\n", result_h);
   result_h = 0.0;
}

checkCudaErrors(cudaGraphExecDestroy(graphExec));
checkCudaErrors(cudaGraphExecDestroy(clonedGraphExec));
checkCudaErrors(cudaGraphDestroy(graph));
checkCudaErrors(cudaGraphDestroy(clonedGraph));
checkCudaErrors(cudaStreamDestroy(streamForGraph));
```

## CUDA NvSci Interop

## CUDA Runtime API
可以关注一下simpleDrvRuntime，演示Driver和Runtime之间的交互
```
#include <cuda.h>
...

CUdevice cuDevice;
CUfunction vecAdd_kernel;
CUmodule cuModule = 0;
CUcontext cuContext;

checkCudaDrvErrors(cuInit(0));
cuDevice = findCudaDevice(argc, (const char **)argv);

// Create context
checkCudaDrvErrors(cuCtxCreate(&cuContext, 0, cuDevice));

// Create module from binary file (FATBIN)
checkCudaDrvErrors(cuModuleLoadData(&cuModule, fatbin.str().c_str()));

// Get function handle from module
checkCudaDrvErrors(cuModuleGetFunction(&vecAdd_kernel, cuModule, "VecAdd_kernel"));

// Runtime API
// Allocate input vectors h_A and h_B in host memory
h_A = (float *)malloc(size);
h_B = (float *)malloc(size);
h_C = (float *)malloc(size);

// Initialize input vectors
RandomInit(h_A, N);
RandomInit(h_B, N);

// Allocate vectors in device memory
checkCudaErrors(cudaMalloc((void**)(&d_A), size));
checkCudaErrors(cudaMalloc((void**)(&d_B), size));
checkCudaErrors(cudaMalloc((void**)(&d_C), size));

// Copy vectors from host memory to device memory
checkCudaErrors(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
checkCudaErrors(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

int threadsPerBlock = 256;
int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

void *args[] = { &d_A, &d_B, &d_C, &N };

// Launch the CUDA kernel
checkCudaDrvErrors(cuLaunchKernel(vecAdd_kernel,  blocksPerGrid, 1, 1,
                       threadsPerBlock, 1, 1,
                       0,
                       stream, args, NULL));

// Copy result from device memory to host memory
// h_C contains the result in host memory
checkCudaErrors(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));
checkCudaErrors(cudaStreamSynchronize(stream));

checkCudaDrvErrors(cuModuleUnload(cuModule));
checkCudaDrvErrors(cuCtxDestroy(cuContext));
```

## CUDA Stream
Stream API definies a sequence of operations that can be overlapped with I/O
## CUDA Stream and Events
Synchronizing Kernels with Event Timers and Streams
## CUDA System Integration
Samples that integrate with Multi Process（ IPC, MPI, OpenMP ）
## CUFFT Library
## CUSolver Library
## CUSparse Library
## Callback Functions
## Cooperative Groups
## Data Parallelelism Algorithm
## Debugging
printf
## Device Memory Allocation
template
```
__global__ void
testKernel(float *g_idata, float *g_odata)
{
    // shared memory
    // the size is determined by the host application
    extern  __shared__  float sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // read in input data from global memory
    sdata[tid] = g_idata[tid];
    __syncthreads();

    // perform some computations
    sdata[tid] = (float) num_threads * sdata[tid];
    __syncthreads();

    // write data to global memory
    g_odata[tid] = sdata[tid];
}
```
## Device Query
## EGLImage-CUDA Interop
## GPU Performance
Samples demonstrating high performance and data I/O
## Graph Analytics
## Graphics Interop
## Image Decoding/Encoding
## Image Processing
## Instantiated CUDA Graph Update
## InterProcess Communication
## Linear Algebra
## MMap
## MPI
## Multi-GPU
## Multithreading
## NVGRAPH, NPP, NVJPEG
## Occupancy Calculator
## Openmp
## Overlap Compute and Copy
## PTX Assembly
## Peer to Peer
## Performance Strategies
## Pinned System Paged Memory
```
    if (bPinGenericMemory)
    {
#if CUDART_VERSION >= 4000
        a_UA = (float *) malloc(bytes + MEMORY_ALIGNMENT);
        b_UA = (float *) malloc(bytes + MEMORY_ALIGNMENT);
        c_UA = (float *) malloc(bytes + MEMORY_ALIGNMENT);

        // We need to ensure memory is aligned to 4K (so we will need to padd memory accordingly)
        a = (float *) ALIGN_UP(a_UA, MEMORY_ALIGNMENT);
        b = (float *) ALIGN_UP(b_UA, MEMORY_ALIGNMENT);
        c = (float *) ALIGN_UP(c_UA, MEMORY_ALIGNMENT);

        checkCudaErrors(cudaHostRegister(a, bytes, cudaHostRegisterMapped));
        checkCudaErrors(cudaHostRegister(b, bytes, cudaHostRegisterMapped));
        checkCudaErrors(cudaHostRegister(c, bytes, cudaHostRegisterMapped));
#endif
    }
    else
    {
#if CUDART_VERSION >= 2020
        flags = cudaHostAllocMapped;
        checkCudaErrors(cudaHostAlloc((void **)&a, bytes, flags));
        checkCudaErrors(cudaHostAlloc((void **)&b, bytes, flags));
        checkCudaErrors(cudaHostAlloc((void **)&c, bytes, flags));
#endif
    }

    /* Initialize the vectors. */

    for (n = 0; n < nelem; n++)
    {
        a[n] = rand() / (float)RAND_MAX;
        b[n] = rand() / (float)RAND_MAX;
    }

    /* Get the device pointers for the pinned CPU memory mapped into the GPU
       memory space. */

#if CUDART_VERSION >= 2020
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_a, (void *)a, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_b, (void *)b, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_c, (void *)c, 0));
#endif
    if (bPinGenericMemory)
    {
#if CUDART_VERSION >= 4000
        checkCudaErrors(cudaHostUnregister(a));
        checkCudaErrors(cudaHostUnregister(b));
        checkCudaErrors(cudaHostUnregister(c));
        free(a_UA);
        free(b_UA);
        free(c_UA);
#endif
```
## Separate Compilation
把device代码放到一个静态库里。deviceFunc。将deviceFunc作为__global__ kernel的参数
```
__global__ void transformVector(float *v, deviceFunc f, uint size)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        v[tid] = (*f)(v[tid]);
    }
}
```
```
// Test library functions.
deviceFunc hFunctionPtr;

cudaMemcpyFromSymbol(&hFunctionPtr,
                     dMultiplyByTwoPtr,
                     sizeof(deviceFunc));
transformVector<<<dimGrid, dimBlock>>>
(dVector, hFunctionPtr, kVectorSize);
checkCudaErrors(cudaGetLastError());
```

## Stream Capture
Create CUDA Jacobi Graph 

## Surface Write
## Unified Memory
## Unified Virtual Address Space (UVA)
## Vote Intrinsics
## cuMemMap IPC


# 高级概念
## 2D Texture
## 3D Graphics
## 3D Texture
## C++11 CUDA
## radix Sort Thrust
```
thrust::host_vector<T> h_keys(numElements);
thrust::host_vector<T> h_keysSorted(numElements);
thrust::host_vector<unsigned int> h_values;

if (!keysOnly)
    h_values = thrust::host_vector<unsigned int>(numElements);

// Fill up with some random data
thrust::default_random_engine rng(clock());

if (floatKeys)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    for (int i = 0; i < (int)numElements; i++)
        h_keys[i] = u01(rng);
}
else
{
    thrust::uniform_int_distribution<unsigned int> u(0, UINT_MAX);

    for (int i = 0; i < (int)numElements; i++)
        h_keys[i] = u(rng);
}

if (!keysOnly)
    thrust::sequence(h_values.begin(), h_values.end());

// Copy data onto the GPU
thrust::device_vector<T> d_keys;
thrust::device_vector<unsigned int> d_values;

// run multiple iterations to compute an average sort time
cudaEvent_t start_event, stop_event;
checkCudaErrors(cudaEventCreate(&start_event));
checkCudaErrors(cudaEventCreate(&stop_event));

float totalTime = 0;

for (unsigned int i = 0; i < numIterations; i++)
{
    // reset data before sort
    d_keys= h_keys;

    if (!keysOnly)
        d_values = h_values;

    checkCudaErrors(cudaEventRecord(start_event, 0));

    if (keysOnly)
        thrust::sort(d_keys.begin(), d_keys.end());
    else
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));

    float time = 0;
    checkCudaErrors(cudaEventElapsedTime(&time, start_event, stop_event));
    totalTime += time;
}

totalTime /= (1.0e3f * numIterations);
printf("radixSortThrust, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements\n",
       1.0e-6f * numElements / totalTime, totalTime, numElements);

getLastCudaError("after radixsort");
```
