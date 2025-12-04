#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

namespace constants
{
    constexpr int BLOCK_SIZE = 32;
}

__global__ void BlockGemmKernel(const float* matrix_a,
                                const float* matrix_b,
                                float* matrix_c,
                                int matrix_size)
{
    int row_idx = blockIdx.y * constants::BLOCK_SIZE + threadIdx.y;
    int col_idx = blockIdx.x * constants::BLOCK_SIZE + threadIdx.x;
    
    if (row_idx >= matrix_size || col_idx >= matrix_size)
    {
        return;
    }
    
    __shared__ float block_a[constants::BLOCK_SIZE][constants::BLOCK_SIZE];
    __shared__ float block_b[constants::BLOCK_SIZE][constants::BLOCK_SIZE];
    
    float thread_result = 0.0f;
    
    int num_blocks = (matrix_size + constants::BLOCK_SIZE - 1) / constants::BLOCK_SIZE;
    
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
    {
        int a_col = block_idx * constants::BLOCK_SIZE + threadIdx.x;
        if (row_idx < matrix_size && a_col < matrix_size)
        {
            block_a[threadIdx.y][threadIdx.x] = matrix_a[row_idx * matrix_size + a_col];
        }
        else
        {
            block_a[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int b_row = block_idx * constants::BLOCK_SIZE + threadIdx.y;
        if (b_row < matrix_size && col_idx < matrix_size)
        {
            block_b[threadIdx.y][threadIdx.x] = matrix_b[b_row * matrix_size + col_idx];
        }
        else
        {
            block_b[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < constants::BLOCK_SIZE; ++k)
        {
            thread_result += block_a[threadIdx.y][k] * block_b[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row_idx < matrix_size && col_idx < matrix_size)
    {
        matrix_c[row_idx * matrix_size + col_idx] = thread_result;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const int total_elements = n * n;
    const size_t data_size_bytes = total_elements * sizeof(float);
    
    std::vector<float> result(total_elements);
    
    float* d_matrix_a = nullptr;
    float* d_matrix_b = nullptr;
    float* d_matrix_c = nullptr;
    
    cudaMalloc(&d_matrix_a, data_size_bytes);
    cudaMalloc(&d_matrix_b, data_size_bytes);
    cudaMalloc(&d_matrix_c, data_size_bytes);
    
    cudaMemcpy(d_matrix_a, a.data(), data_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, b.data(), data_size_bytes, cudaMemcpyHostToDevice);
    
    dim3 threads_per_block(constants::BLOCK_SIZE, constants::BLOCK_SIZE);
    int blocks_in_grid = (n + constants::BLOCK_SIZE - 1) / constants::BLOCK_SIZE;
    dim3 blocks_per_grid(blocks_in_grid, blocks_in_grid);
    
    BlockGemmKernel<<<blocks_per_grid, threads_per_block>>>(d_matrix_a,
                                                            d_matrix_b,
                                                            d_matrix_c,
                                                            n);
    
    cudaMemcpy(result.data(), d_matrix_c, data_size_bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);
    
    return result;
}