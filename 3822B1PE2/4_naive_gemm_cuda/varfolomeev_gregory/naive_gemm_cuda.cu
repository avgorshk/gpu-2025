#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define UNROLL_FACTOR 4

__global__ void transpose_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        output[col * n + row] = input[row * n + col];
    }
}

__global__ void naive_gemm_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b_t,
                                   float* __restrict__ c,
                                   int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        
        int k = 0;
        int n_unrolled = (n / UNROLL_FACTOR) * UNROLL_FACTOR;
        
        // Loop unrolling for better instruction-level parallelism
        for (; k < n_unrolled; k += UNROLL_FACTOR) {
            sum += a[row * n + k] * b_t[col * n + k];
            sum += a[row * n + k + 1] * b_t[col * n + k + 1];
            sum += a[row * n + k + 2] * b_t[col * n + k + 2];
            sum += a[row * n + k + 3] * b_t[col * n + k + 3];
        }
        
        // Handle remaining elements
        for (; k < n; ++k) {
            sum += a[row * n + k] * b_t[col * n + k];
        }
        
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }
    
    size_t matrix_size = static_cast<size_t>(n) * n;
    size_t bytes = matrix_size * sizeof(float);
    
    if (a.size() != matrix_size || b.size() != matrix_size) {
        return {};
    }
    
    std::vector<float> c(matrix_size);
    
    float *d_a, *d_b, *d_b_t, *d_c;
    
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_b_t, bytes);
    cudaMalloc(&d_c, bytes);
    
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    transpose_kernel<<<grid_dim, block_dim>>>(d_b, d_b_t, n);
    
    naive_gemm_kernel<<<grid_dim, block_dim>>>(d_a, d_b_t, d_c, n);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_b_t);
    cudaFree(d_c);
    
    return c;
}

