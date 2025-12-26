#include "naive_gemm_cuda.h"
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

__global__ void naiveGemmKernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n && col >= n) return;

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> result(n * n);
    
    float *deviceA, *deviceB, *deviceResult;
    size_t bytes = n * n * sizeof(float);
    
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceResult, bytes);
    
    cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice);
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    
    int sqrtMax = static_cast<int>(sqrtf(maxThreadsPerBlock));
    int blockSize = 1;
    while (blockSize * 2 <= sqrtMax) {
        blockSize <<= 1;
    }
    
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    naiveGemmKernel<<<numBlocks, threadsPerBlock>>>(deviceA, deviceB, deviceResult, n);
    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), deviceResult, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceResult);
    
    return result;
}