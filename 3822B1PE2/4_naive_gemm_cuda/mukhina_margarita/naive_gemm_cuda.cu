#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

__global__ void gemm_kernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        
        int k = 0;
        for (; k <= n - 4; k += 4) {
            sum += A[row * n + k] * B[k * n + col] +
                   A[row * n + k + 1] * B[(k + 1) * n + col] +
                   A[row * n + k + 2] * B[(k + 2) * n + col] +
                   A[row * n + k + 3] * B[(k + 3) * n + col];
        }
        
        for (; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        
        C[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n);
    
    float *d_a, *d_b, *d_c;
    
    cudaMalloc(&d_a, n * n * sizeof(float));
    cudaMalloc(&d_b, n * n * sizeof(float));
    cudaMalloc(&d_c, n * n * sizeof(float));
    
    cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((n + 15) / 16, (n + 15) / 16);
    
    gemm_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}