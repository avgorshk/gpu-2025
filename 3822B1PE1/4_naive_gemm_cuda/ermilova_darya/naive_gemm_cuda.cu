#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void cudaCheck(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(msg) +
            ": " + cudaGetErrorString(err));
}

__global__ void NaiveGemmKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= n || col >= n)
        return;

    float sum = 0.0f;
    int rowOffset = row * n;

    int k = 0;

#pragma unroll
    for (; k + 3 < n; k += 4)
    {
        float a0 = A[rowOffset + k];
        float a1 = A[rowOffset + k + 1];
        float a2 = A[rowOffset + k + 2];
        float a3 = A[rowOffset + k + 3];

        float b0 = B[(k)*n + col];
        float b1 = B[(k + 1) * n + col];
        float b2 = B[(k + 2) * n + col];
        float b3 = B[(k + 3) * n + col];

        sum += a0 * b0;
        sum += a1 * b1;
        sum += a2 * b2;
        sum += a3 * b3;
    }

    for (; k < n; ++k)
    {
        sum += A[rowOffset + k] * B[k * n + col];
    }

    C[rowOffset + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n)
{
    const std::size_t expected_size = (std::size_t)n * n;

    if (a.size() != expected_size || b.size() != expected_size)
        throw std::runtime_error("NaiveGemmCUDA: wrong matrix size");

    std::size_t bytes = expected_size * sizeof(float);

    float* dA = nullptr, * dB = nullptr, * dC = nullptr;

    cudaCheck(cudaMalloc(&dA, bytes), "cudaMalloc dA failed");
    cudaCheck(cudaMalloc(&dB, bytes), "cudaMalloc dB failed");
    cudaCheck(cudaMalloc(&dC, bytes), "cudaMalloc dC failed");

    cudaCheck(cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy H2D A failed");
    cudaCheck(cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy H2D B failed");

    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x,
        (n + block.y - 1) / block.y);

    NaiveGemmKernel << <grid, block >> > (dA, dB, dC, n);
    cudaCheck(cudaGetLastError(), "Kernel launch failed");
    cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    std::vector<float> c(expected_size);
    cudaCheck(cudaMemcpy(c.data(), dC, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H C failed");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return c;
}
