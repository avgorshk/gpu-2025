#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>

constexpr int TILE_SIZE = 16;

__global__ void gemm_block_kernel(const float *__restrict__ a,
                                  const float *__restrict__ b,
                                  float *__restrict__ c,
                                  int n)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float c_val = 0.0f;

    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {
        int a_row = row;
        int a_col = tile * TILE_SIZE + tx;
        if (a_row < n && a_col < n)
        {
            As[ty][tx] = a[a_row * n + a_col];
        }
        else
        {
            As[ty][tx] = 0.0f;
        }

        int b_row = tile * TILE_SIZE + ty;
        int b_col = col;
        if (b_row < n && b_col < n)
        {
            Bs[ty][tx] = b[b_row * n + b_col];
        }
        else
        {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
        {
            c_val += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n)
    {
        c[row * n + col] = c_val;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    if (n == 0)
        return std::vector<float>();

    size_t bytes = n * n * sizeof(float);
    std::vector<float> c(n * n, 0.0f);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    gemm_block_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
    }

    return c;
}