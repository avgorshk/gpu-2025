#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for naive matrix multiplication
// Each thread computes one element C[i][j]
__global__ void naive_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        // Unroll loop for better performance
        #pragma unroll
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n == 0) {
        return std::vector<float>();
    }
    
    // Allocate device memory
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    size_t bytes = n * n * sizeof(float);
    
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy input matrices to device
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    // Use 16x16 thread blocks for warp-friendly access
    const int blockSize = 16;
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);
    
    // Launch kernel
    naive_gemm_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
    // Allocate host output and copy result back
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}

