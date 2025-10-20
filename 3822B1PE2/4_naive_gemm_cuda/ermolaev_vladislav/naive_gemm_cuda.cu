#include "naive_gemm_cuda.h"
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

__global__ void naiveGemmKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n && j >= n) return;

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[i * n + k] * b[k * n + j];
    }
    c[i * n + j] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n);
    
    float *d_a, *d_b, *d_c;
    size_t size = n * n * sizeof(float);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    
    int sqrtMax = static_cast<int>(sqrtf(maxThreadsPerBlock));
    int blockSize = 1;
    while (blockSize * 2 <= sqrtMax) {
        blockSize <<= 1;
    }
    
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    naiveGemmKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}