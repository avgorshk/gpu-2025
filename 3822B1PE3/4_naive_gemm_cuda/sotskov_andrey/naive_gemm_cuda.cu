#include <cstdlib>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "naive_gemm_cuda.h"

#define BLOCK_SIZE 32

__global__ void NaiveGemmKernel(const float* matrix_a, const float* matrix_b, float* matrix_c,
                                const size_t matrix_size)
{
    constexpr int tile_size = BLOCK_SIZE;
    __shared__ float tile_a[tile_size][tile_size];
    __shared__ float tile_b[tile_size][tile_size];

    size_t row_index = blockIdx.y * tile_size + threadIdx.y;
    size_t column_index = blockIdx.x * tile_size + threadIdx.x;

    float accumulator = 0.0f;

    for (size_t tile_offset = 0; tile_offset < matrix_size; tile_offset += tile_size)
    {
        if (column_index < matrix_size && (threadIdx.y + tile_offset) < matrix_size)
        {
            tile_b[threadIdx.y][threadIdx.x] = matrix_b[(threadIdx.y + tile_offset) * matrix_size + column_index];
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (row_index < matrix_size && (threadIdx.x + tile_offset) < matrix_size)
        {
            tile_a[threadIdx.y][threadIdx.x] = matrix_a[row_index * matrix_size + threadIdx.x + tile_offset];
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (size_t element_index = 0; element_index < tile_size; ++element_index)
        {
            accumulator += tile_a[threadIdx.y][element_index] * tile_b[element_index][threadIdx.x];
        }

        __syncthreads();
    }

    if (row_index < matrix_size && column_index < matrix_size)
    {
        matrix_c[row_index * matrix_size + column_index] = accumulator;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& matrix_a,
                                 const std::vector<float>& matrix_b,
                                 int matrix_size)
{
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);

    size_t total_elements = matrix_size * matrix_size;
    std::vector<float> result_matrix(total_elements);
    size_t total_bytes = total_elements * sizeof(float);

    dim3 thread_block_dimensions(BLOCK_SIZE, BLOCK_SIZE);
    size_t blocks_per_dimension = (matrix_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid_dimensions(blocks_per_dimension, blocks_per_dimension);

    float* device_matrix_a = nullptr;
    float* device_matrix_b = nullptr;
    float* device_matrix_c = nullptr;

    cudaMalloc(&device_matrix_a, total_bytes);
    cudaMalloc(&device_matrix_b, total_bytes);
    cudaMalloc(&device_matrix_c, total_bytes);

    cudaMemcpy(device_matrix_a, matrix_a.data(), total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix_b, matrix_b.data(), total_bytes, cudaMemcpyHostToDevice);

    NaiveGemmKernel<<<grid_dimensions, thread_block_dimensions>>>(device_matrix_a, device_matrix_b, device_matrix_c, matrix_size);

    cudaDeviceSynchronize();
    cudaMemcpy(result_matrix.data(), device_matrix_c, total_bytes, cudaMemcpyDeviceToHost);

    cudaFree(device_matrix_c);
    cudaFree(device_matrix_b);
    cudaFree(device_matrix_a);

    return result_matrix;
}