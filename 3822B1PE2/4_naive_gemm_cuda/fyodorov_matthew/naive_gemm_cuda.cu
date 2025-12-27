#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <algorithm>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

__global__ void naive_gemm_kernel_basic(const float *A, const float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void naive_gemm_kernel_optimized(const float *A, const float *B, float *C, int n)
{

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < n; tile += TILE_SIZE)
    {
        if (row < n && (tile + tx) < n)
        {
            As[ty][tx] = A[row * n + (tile + tx)];
        }
        else
        {
            As[ty][tx] = 0.0f;
        }

        if ((tile + ty) < n && col < n)
        {
            Bs[ty][tx] = B[(tile + ty) * n + col];
        }
        else
        {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n)
    {
        C[row * n + col] = sum;
    }
}

__global__ void naive_gemm_kernel_vectorized(const float *A, const float *B, float *C, int n)
{

    const int elements_per_thread = 4;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_start = blockIdx.x * (blockDim.x * elements_per_thread) + threadIdx.x;

    if (row < n)
    {
#pragma unroll
        for (int e = 0; e < elements_per_thread; ++e)
        {
            int col = col_start + e * blockDim.x;
            if (col < n)
            {
                float sum = 0.0f;

                int k = 0;
                for (; k <= n - 4; k += 4)
                {
                    sum += A[row * n + k] * B[k * n + col] + A[row * n + k + 1] * B[(k + 1) * n + col] + A[row * n + k + 2] * B[(k + 2) * n + col] + A[row * n + k + 3] * B[(k + 3) * n + col];
                }

                for (; k < n; ++k)
                {
                    sum += A[row * n + k] * B[k * n + col];
                }

                C[row * n + col] = sum;
            }
        }
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    std::vector<float> c(n * n, 0.0f);

    if (a.size() != n * n || b.size() != n * n)
    {
        std::cerr << "Ошибка: неверный размер входных матриц" << std::endl;
        return c;
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    size_t matrix_size = n * n * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&d_a, matrix_size);
    if (err != cudaSuccess)
    {
        std::cerr << "Ошибка cudaMalloc (d_a): " << cudaGetErrorString(err) << std::endl;
        return c;
    }

    err = cudaMalloc(&d_b, matrix_size);
    if (err != cudaSuccess)
    {
        std::cerr << "Ошибка cudaMalloc (d_b): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a);
        return c;
    }

    err = cudaMalloc(&d_c, matrix_size);
    if (err != cudaSuccess)
    {
        std::cerr << "Ошибка cudaMalloc (d_c): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        return c;
    }

    cudaMemcpyAsync(d_a, a.data(), matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, b.data(), matrix_size, cudaMemcpyHostToDevice);

    dim3 block_dim, grid_dim;

    if (n <= 256)
    {
        block_dim = dim3(16, 16);
        grid_dim = dim3((n + 15) / 16, (n + 15) / 16);

        naive_gemm_kernel_basic<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
    }
    else if (n <= 1024)
    {
        block_dim = dim3(TILE_SIZE, TILE_SIZE);
        grid_dim = dim3((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

        naive_gemm_kernel_optimized<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
    }
    else
    {
        const int threads_per_block = 256;
        const int elements_per_thread = 4;

        block_dim = dim3(threads_per_block, 1);
        grid_dim = dim3((n + (threads_per_block * elements_per_thread) - 1) /
                            (threads_per_block * elements_per_thread),
                        (n + 0) / 1);

        naive_gemm_kernel_vectorized<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Ошибка запуска ядра CUDA: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(c.data(), d_c, matrix_size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}