#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t bytes = n * n * sizeof(float);
    
    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static int allocated_n = 0;
    
    if (allocated_n < n) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        allocated_n = n;
    }
    
    std::vector<float> c(n * n);
    
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);
    
    gemm_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    
    return c;
}
