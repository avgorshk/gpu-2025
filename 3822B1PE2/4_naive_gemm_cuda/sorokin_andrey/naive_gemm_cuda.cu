#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

__global__ void matrix_transpose(const float* __restrict__ source,
    float* __restrict__ destination,
    int matrix_size) {
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (y_index < matrix_size && x_index < matrix_size) {
        destination[x_index * matrix_size + y_index] = source[y_index * matrix_size + x_index];
    }
}

__global__ void simple_matrix_multiply(const float* __restrict__ matrix_a,
    const float* __restrict__ transposed_matrix_b,
    float* __restrict__ result_matrix, int matrix_size) {
    int y_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int x_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (y_pos < matrix_size && x_pos < matrix_size) {
        float accumulator = 0.0f;
        for (int inner_idx = 0; inner_idx < matrix_size; inner_idx++) {
            accumulator += matrix_a[y_pos * matrix_size + inner_idx] * 
                          transposed_matrix_b[x_pos * matrix_size + inner_idx];
        }
        result_matrix[y_pos * matrix_size + x_pos] = accumulator;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("Matrix sizes do not match n*n");
    }
    size_t memory_size = n * n * sizeof(float);
    std::vector<float> result(n * n);
    if (n == 0) return result;

    float* device_matrix_a = nullptr;
    float* device_matrix_b = nullptr;
    float* device_transposed_b = nullptr;
    float* device_result = nullptr;

    cudaMalloc(&device_matrix_a, memory_size);
    cudaMalloc(&device_matrix_b, memory_size);
    cudaMalloc(&device_transposed_b, memory_size);
    cudaMalloc(&device_result, memory_size);
    cudaMemcpy(device_matrix_a, a.data(), memory_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix_b, b.data(), memory_size, cudaMemcpyHostToDevice);

    dim3 thread_block_transpose(16, 16);
    dim3 grid_dim_transpose((n + 16 - 1) / 16,
                           (n + 16 - 1) / 16);

    matrix_transpose << <grid_dim_transpose, thread_block_transpose >> > 
        (device_matrix_b, device_transposed_b, n);
    cudaDeviceSynchronize();

    dim3 thread_block_multiply(16, 16);
    dim3 grid_dim_multiply((n + 16 - 1) / 16,
                          (n + 16 - 1) / 16);

    simple_matrix_multiply << <grid_dim_multiply, thread_block_multiply >> > 
        (device_matrix_a, device_transposed_b, device_result, n);
    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), device_result, memory_size, cudaMemcpyDeviceToHost);

    cudaFree(device_matrix_a);
    cudaFree(device_matrix_b);
    cudaFree(device_transposed_b);
    cudaFree(device_result);

    return result;
}