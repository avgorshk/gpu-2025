#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

__global__ void matrix_multiply_kernel(const float *__restrict__ input_matrix_a,
                                       const float *__restrict__ input_matrix_b,
                                       float *__restrict__ output_matrix_c,
                                       int matrix_size)
{
    __shared__ float tile_a[16][16];
    __shared__ float tile_b[16][16];
    __shared__ float tile_c[16][16];

    int tile_index_y = blockIdx.y;
    int tile_index_x = blockIdx.x;
    int thread_index_y = threadIdx.y;
    int thread_index_x = threadIdx.x;

    int global_index_y = tile_index_y * 16 + thread_index_y;
    int global_index_x = tile_index_x * 16 + thread_index_x;

    tile_c[thread_index_y][thread_index_x] = 0.0f;

    int total_tiles = matrix_size / 16;

    for (int tile_num = 0; tile_num < total_tiles; ++tile_num)
    {
        int a_row = tile_index_y * 16 + thread_index_y;
        int a_col = tile_num * 16 + thread_index_x;
        tile_a[thread_index_y][thread_index_x] = input_matrix_a[a_row * matrix_size + a_col];

        int b_row = tile_num * 16 + thread_index_y;
        int b_col = tile_index_x * 16 + thread_index_x;
        tile_b[thread_index_y][thread_index_x] = input_matrix_b[b_row * matrix_size + b_col];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < 16; ++k)
        {
            tile_c[thread_index_y][thread_index_x] += tile_a[thread_index_y][k] * tile_b[k][thread_index_x];
        }

        __syncthreads();
    }

    if (global_index_y < matrix_size && global_index_x < matrix_size)
    {
        output_matrix_c[global_index_y * matrix_size + global_index_x] = tile_c[thread_index_y][thread_index_x];
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    if ((n & (n - 1)) != 0)
    {
        throw std::invalid_argument("power of 2");
    }

    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n))
    {
        throw std::invalid_argument("not n*n");
    }

    std::vector<float> result(n * n);

    if (n == 0)
        return result;

    float *device_a, *device_b, *device_c;
    size_t memory_size = n * n * sizeof(float);

    cudaMalloc(&device_a, memory_size);
    cudaMalloc(&device_b, memory_size);
    cudaMalloc(&device_c, memory_size);

    cudaMemcpy(device_a, a.data(), memory_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b.data(), memory_size, cudaMemcpyHostToDevice);

    dim3 block_config(16, 16);
    dim3 grid_config(n / 16, n / 16);

    matrix_multiply_kernel<<<grid_config, block_config>>>(device_a, device_b, device_c, n);

    cudaMemcpy(result.data(), device_c, memory_size, cudaMemcpyDeviceToHost);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return result;
}