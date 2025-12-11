#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void basic_matmul_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int n)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        const int a_offset = row * n;

#pragma unroll 
        for (int k = 0; k < n; ++k)
        {
            sum += A[a_offset + k] * B[k * n + col];
        }

        C[a_offset + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = count * sizeof(float);

    static float* bufA = nullptr;
    static float* bufB = nullptr;
    static float* bufC = nullptr;
    static size_t capacity = 0;

    if (capacity < count)
    {
        if (bufA) cudaFree(bufA);
        if (bufB) cudaFree(bufB);
        if (bufC) cudaFree(bufC);

        cudaMalloc(&bufA, bytes);
        cudaMalloc(&bufB, bytes);
        cudaMalloc(&bufC, bytes);

        capacity = count;
    }

    cudaMemcpy(bufA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bufB, b.data(), bytes, cudaMemcpyHostToDevice);

    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE,
              (n + TILE - 1) / TILE);

    basic_matmul_kernel<<<grid, block>>>(bufA, bufB, bufC, n);
    cudaDeviceSynchronize();

    std::vector<float> result(count);
    cudaMemcpy(result.data(), bufC, bytes, cudaMemcpyDeviceToHost);

    return result;
}
