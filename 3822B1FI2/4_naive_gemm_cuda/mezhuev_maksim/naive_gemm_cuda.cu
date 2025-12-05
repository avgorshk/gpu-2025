#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

#define BLOCK_DIM 32

namespace {
    __global__ void MatrixMultiplyKernel(
        const float* mat_A,
        const float* mat_B,
        float* mat_C,
        int matrix_dim) {
        
        __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
        __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];
        
        int global_row = blockIdx.y * BLOCK_DIM + threadIdx.y;
        int global_col = blockIdx.x * BLOCK_DIM + threadIdx.x;
        float accumulator = 0.0f;
        
        for (int tile_idx = 0; tile_idx < matrix_dim; tile_idx += BLOCK_DIM) {
            bool load_A_valid = (global_row < matrix_dim) && 
                                (tile_idx + threadIdx.x < matrix_dim);
            bool load_B_valid = (global_col < matrix_dim) && 
                                (tile_idx + threadIdx.y < matrix_dim);
            
            shared_A[threadIdx.y][threadIdx.x] = load_A_valid ?
                mat_A[global_row * matrix_dim + tile_idx + threadIdx.x] : 0.0f;
            
            shared_B[threadIdx.y][threadIdx.x] = load_B_valid ?
                mat_B[(tile_idx + threadIdx.y) * matrix_dim + global_col] : 0.0f;
            
            __syncthreads();
            
            #pragma unroll
            for (int inner_idx = 0; inner_idx < BLOCK_DIM; ++inner_idx) {
                accumulator += shared_A[threadIdx.y][inner_idx] * 
                              shared_B[inner_idx][threadIdx.x];
            }
            
            __syncthreads();
        }
        
        if (global_row < matrix_dim && global_col < matrix_dim) {
            mat_C[global_row * matrix_dim + global_col] = accumulator;
        }
    }
}

std::vector<float> CudaMatrixMultiply(
    const std::vector<float>& matrix_A,
    const std::vector<float>& matrix_B,
    int matrix_dim) {
    
    size_t total_bytes = matrix_dim * matrix_dim * sizeof(float);
    float *dev_A, *dev_B, *dev_C;
    
    cudaMalloc(&dev_A, total_bytes);
    cudaMalloc(&dev_B, total_bytes);
    cudaMalloc(&dev_C, total_bytes);
    
    cudaMemcpy(dev_A, matrix_A.data(), total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, matrix_B.data(), total_bytes, cudaMemcpyHostToDevice);
    
    dim3 thread_block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_blocks(
        (matrix_dim + BLOCK_DIM - 1) / BLOCK_DIM,
        (matrix_dim + BLOCK_DIM - 1) / BLOCK_DIM
    );
    
    MatrixMultiplyKernel<<<grid_blocks, thread_block>>>(
        dev_A, dev_B, dev_C, matrix_dim);
    
    cudaDeviceSynchronize();
    
    std::vector<float> result(matrix_dim * matrix_dim);
    cudaMemcpy(result.data(), dev_C, total_bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    
    return result;
}