#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

__global__ void naive_gemm_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += a[i * n + k] * b[k * n + j];
        }
        c[i * n + j] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n, 0.0f);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * n * sizeof(float));
    cudaMalloc(&d_b, n * n * sizeof(float));
    cudaMalloc(&d_c, n * n * sizeof(float));
    
    cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockSize(32, 8);
    dim3 numBlocks((n + blockSize.x - 1) / blockSize.x, 
                  (n + blockSize.y - 1) / blockSize.y);
    
    naive_gemm_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}