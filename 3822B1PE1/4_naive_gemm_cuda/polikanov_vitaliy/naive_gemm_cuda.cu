#include <iostream>
#include <vector>
#include <stdexcept>
#include "naive_gemm_cuda.h"

__global__ void matmul_naive_kernel(const float* __restrict__ left,
                                    const float* __restrict__ right,
                                    float* __restrict__ output,
                                    int dim) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; // строка
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; // столбец
    if (i < dim && j < dim) {
        float accumulator = 0.0f;
        for (int t = 0; t < dim; ++t) {
            accumulator += left[i * dim + t] * right[t * dim + j];
        }
        output[i * dim + j] = accumulator;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& lhs,
                                 const std::vector<float>& rhs,
                                 int dim) {
    const auto total_elements = static_cast<size_t>(dim) * dim;
    const auto memory_bytes = total_elements * sizeof(float);
    float *dev_lhs = nullptr, *dev_rhs = nullptr, *dev_out = nullptr;
    cudaMalloc(&dev_lhs, memory_bytes);
    cudaMalloc(&dev_rhs, memory_bytes);
    cudaMalloc(&dev_out, memory_bytes);
    cudaMemcpy(dev_lhs, lhs.data(), memory_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_rhs, rhs.data(), memory_bytes, cudaMemcpyHostToDevice);

    constexpr int TILE = 16;
    dim3 block_shape(TILE, TILE);
    dim3 grid_shape(
            (dim + TILE - 1) / TILE,
            (dim + TILE - 1) / TILE
    );
    matmul_naive_kernel<<<grid_shape, block_shape>>>(dev_lhs, dev_rhs, dev_out, dim);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::cerr << "Ошибка при запуске CUDA-ядра: "
                  << cudaGetErrorString(launch_err) << std::endl;
        cudaFree(dev_lhs);
        cudaFree(dev_rhs);
        cudaFree(dev_out);
        throw std::runtime_error("Не удалось выполнить CUDA-ядро");
    }
    std::vector<float> result(total_elements);
    cudaMemcpy(result.data(), dev_out, memory_bytes, cudaMemcpyDeviceToHost);
    cudaFree(dev_lhs);
    cudaFree(dev_rhs);
    cudaFree(dev_out);
    return result;
}