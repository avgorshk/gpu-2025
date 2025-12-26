#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void block_gemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int n) {
    // Shared memory for tiles with padding to avoid bank conflicts
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE + 1];
    
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    
    int global_row = block_row * BLOCK_SIZE + thread_row;
    int global_col = block_col * BLOCK_SIZE + thread_col;
    
    float sum = 0.0f;
    
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Loop over blocks
    for (int block_k = 0; block_k < num_blocks; ++block_k) {
        // Load tile A into shared memory (coalesced access)
        int a_col = block_k * BLOCK_SIZE + thread_col;
        if (global_row < n && a_col < n) {
            sA[thread_row][thread_col] = A[global_row * n + a_col];
        } else {
            sA[thread_row][thread_col] = 0.0f;
        }
        
        // Load tile B into shared memory (coalesced access)
        int b_row = block_k * BLOCK_SIZE + thread_row;
        if (b_row < n && global_col < n) {
            sB[thread_row][thread_col] = B[b_row * n + global_col];
        } else {
            sB[thread_row][thread_col] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product with loop unrolling
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[thread_row][k] * sB[k][thread_col];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    if (global_row < n && global_col < n) {
        C[global_row * n + global_col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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
    
    float *d_a, *d_b, *d_c;
    
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    block_gemm_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}

