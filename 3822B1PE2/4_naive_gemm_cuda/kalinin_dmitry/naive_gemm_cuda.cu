#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void naive_gemm_kernel_optimized(const float* __restrict__ a,
                                           const float* __restrict__ b,
                                           float* __restrict__ c,
                                           int n) {
    int base_row = blockIdx.y * blockDim.y;
    int base_col = blockIdx.x * blockDim.x;
    int row = base_row + threadIdx.y;
    int col = base_col + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        const float* a_row = &a[row * n];
        const float* b_col = &b[col];
        
        int k = 0;
        for (; k < n - 7; k += 8) {
            sum += a_row[k] * b_col[k * n];
            sum += a_row[k + 1] * b_col[(k + 1) * n];
            sum += a_row[k + 2] * b_col[(k + 2) * n];
            sum += a_row[k + 3] * b_col[(k + 3) * n];
            sum += a_row[k + 4] * b_col[(k + 4) * n];
            sum += a_row[k + 5] * b_col[(k + 5) * n];
            sum += a_row[k + 6] * b_col[(k + 6) * n];
            sum += a_row[k + 7] * b_col[(k + 7) * n];
        }
        for (; k < n; ++k) {
            sum += a_row[k] * b_col[k * n];
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
    
    dim3 blockSize(32, 32);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (n + blockSize.y - 1) / blockSize.y);
    
    naive_gemm_kernel_optimized<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}

