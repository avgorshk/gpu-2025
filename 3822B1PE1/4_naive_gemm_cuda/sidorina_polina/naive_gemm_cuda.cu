#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

__global__ void NaiveGemmKernel(const float* matrix_a,
                                const float* matrix_b,
                                float* matrix_c,
                                int matrix_size)
{
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row_idx < matrix_size && col_idx < matrix_size)
    {
        float dot_product = 0.0f;
        
        for (int k = 0; k < matrix_size; ++k)
        {
            float a_element = matrix_a[row_idx * matrix_size + k];
            float b_element = matrix_b[k * matrix_size + col_idx];
            dot_product += a_element * b_element;
        }
        
        matrix_c[row_idx * matrix_size + col_idx] = dot_product;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    constexpr int THREADS_PER_DIM = 16;
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

    dim3 threads_per_block(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 blocks_per_grid((n + THREADS_PER_DIM - 1) / THREADS_PER_DIM,
                         (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM);

    NaiveGemmKernel<<<blocks_per_grid, threads_per_block>>>(d_matrix_a, 
                                                             d_matrix_b, 
                                                             d_matrix_c, 
                                                             n);
    
    cudaMemcpy(result.data(), d_matrix_c, data_size_bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);
    
    return result;
}