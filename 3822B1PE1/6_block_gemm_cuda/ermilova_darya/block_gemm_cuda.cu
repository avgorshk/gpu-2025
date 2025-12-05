#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define BLOCK_TILE 16


inline void cudaCheck(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(msg) +
            ": " + cudaGetErrorString(err));
}

__global__ void BlockGemmKernel(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int n)
{
    int row = blockIdx.y * BLOCK_TILE + threadIdx.y;
    int col = blockIdx.x * BLOCK_TILE + threadIdx.x;

    __shared__ float As[BLOCK_TILE][BLOCK_TILE];
    __shared__ float Bs[BLOCK_TILE][BLOCK_TILE];

    float acc = 0.0f;
    int numTiles = (n + BLOCK_TILE - 1) / BLOCK_TILE;

    for (int t = 0; t < numTiles; ++t)
    {
        int aCol = t * BLOCK_TILE + threadIdx.x;
        int bRow = t * BLOCK_TILE + threadIdx.y;

        if (row < n && aCol < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < n && col < n)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_TILE; ++k)
        {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }
    if (row < n && col < n)
        C[row * n + col] = acc;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n)
{
    const std::size_t expected_size = static_cast<std::size_t>(n) * n;
    if (a.size() != expected_size || b.size() != expected_size)
        throw std::runtime_error("BlockGemmCUDA: wrong matrix size");

    std::size_t bytes = expected_size * sizeof(float);

    float* dA = nullptr, * dB = nullptr, * dC = nullptr;

    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream), "cudaStreamCreate failed");

    cudaCheck(cudaMalloc(&dA, bytes), "cudaMalloc dA failed");
    cudaCheck(cudaMalloc(&dB, bytes), "cudaMalloc dB failed");
    cudaCheck(cudaMalloc(&dC, bytes), "cudaMalloc dC failed");

    cudaCheck(cudaMemcpyAsync(dA, a.data(), bytes,
        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync A H2D failed");
    cudaCheck(cudaMemcpyAsync(dB, b.data(), bytes,
        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync B H2D failed");

    dim3 block(BLOCK_TILE, BLOCK_TILE);
    dim3 grid((n + BLOCK_TILE - 1) / BLOCK_TILE,
        (n + BLOCK_TILE - 1) / BLOCK_TILE);

    BlockGemmKernel << <grid, block, 0, stream >> > (dA, dB, dC, n);
    cudaCheck(cudaGetLastError(), "BlockGemmKernel launch failed");

    std::vector<float> c(expected_size);
    cudaCheck(cudaMemcpyAsync(c.data(), dC, bytes,
        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync C D2H failed");

    cudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");
    cudaCheck(cudaStreamDestroy(stream), "cudaStreamDestroy failed");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return c;
}
