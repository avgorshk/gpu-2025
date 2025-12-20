#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void naive_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        int k = 0;
        for (; k < n - 3; k += 4) {
            sum += a[row * n + k] * b[k * n + col];
            sum += a[row * n + k + 1] * b[(k + 1) * n + col];
            sum += a[row * n + k + 2] * b[(k + 2) * n + col];
            sum += a[row * n + k + 3] * b[(k + 3) * n + col];
        }
        for (; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t size = n * n * sizeof(float);
    std::vector<float> c(n * n);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (n + blockSize.y - 1) / blockSize.y);
    
    naive_gemm_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}

