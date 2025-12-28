#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <algorithm>

#define BLOCK_SIZE 16
#define TILE_SIZE 32

__global__ void block_gemm_kernel(const float *A, const float *B, float *C, int n)
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int row = blockRow * BLOCK_SIZE + threadRow;
    int col = blockCol * BLOCK_SIZE + threadCol;

    float sum = 0.0f;

    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < numTiles; ++tile)
    {
        int A_row = row;
        int A_col = tile * BLOCK_SIZE + threadCol;
        if (A_row < n && A_col < n)
        {
            As[threadRow][threadCol] = A[A_row * n + A_col];
        }
        else
        {
            As[threadRow][threadCol] = 0.0f;
        }

        int B_row = tile * BLOCK_SIZE + threadRow;
        int B_col = col;
        if (B_row < n && B_col < n)
        {
            Bs[threadRow][threadCol] = B[B_row * n + B_col];
        }
        else
        {
            Bs[threadRow][threadCol] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += As[threadRow][k] * Bs[k][threadCol];
        }

        __syncthreads();
    }

    if (row < n && col < n)
    {
        C[row * n + col] = sum;
    }
}

__global__ void block_gemm_kernel_optimized(const float *A, const float *B, float *C, int n)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int elements_per_thread = 4;

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int threads_per_dim = TILE_SIZE / elements_per_thread;

    int startRow = blockRow * TILE_SIZE + threadRow * elements_per_thread;
    int startCol = blockCol * TILE_SIZE + threadCol * elements_per_thread;

    float sum[elements_per_thread][elements_per_thread];
    for (int i = 0; i < elements_per_thread; ++i)
    {
        for (int j = 0; j < elements_per_thread; ++j)
        {
            sum[i][j] = 0.0f;
        }
    }

    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile)
    {
        for (int i = 0; i < elements_per_thread; ++i)
        {
            int loadRow = threadRow * elements_per_thread + i;
            for (int j = 0; j < elements_per_thread; ++j)
            {
                int loadCol = threadCol * elements_per_thread + j;
                int A_row = blockRow * TILE_SIZE + loadRow;
                int A_col = tile * TILE_SIZE + loadCol;

                if (A_row < n && A_col < n)
                {
                    As[loadRow][loadCol] = A[A_row * n + A_col];
                }
                else
                {
                    As[loadRow][loadCol] = 0.0f;
                }
            }
        }

        for (int i = 0; i < elements_per_thread; ++i)
        {
            int loadRow = threadRow * elements_per_thread + i;
            for (int j = 0; j < elements_per_thread; ++j)
            {
                int loadCol = threadCol * elements_per_thread + j;
                int B_row = tile * TILE_SIZE + loadRow;
                int B_col = blockCol * TILE_SIZE + loadCol;

                if (B_row < n && B_col < n)
                {
                    Bs[loadRow][loadCol] = B[B_row * n + B_col];
                }
                else
                {
                    Bs[loadRow][loadCol] = 0.0f;
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            for (int i = 0; i < elements_per_thread; ++i)
            {
                float a_val = As[threadRow * elements_per_thread + i][k];
                for (int j = 0; j < elements_per_thread; ++j)
                {
                    float b_val = Bs[k][threadCol * elements_per_thread + j];
                    sum[i][j] += a_val * b_val;
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < elements_per_thread; ++i)
    {
        int row = startRow + i;
        if (row < n)
        {
            for (int j = 0; j < elements_per_thread; ++j)
            {
                int col = startCol + j;
                if (col < n)
                {
                    C[row * n + col] = sum[i][j];
                }
            }
        }
    }
}

__global__ void block_gemm_kernel_small(const float *A, const float *B, float *C, int n)
{

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile)
    {
        int A_row = row;
        int A_col = tile * BLOCK_SIZE + tx;
        if (A_row < n && A_col < n)
        {
            As[ty][tx] = A[A_row * n + A_col];
        }
        else
        {
            As[ty][tx] = 0.0f;
        }

        int B_row = tile * BLOCK_SIZE + ty;
        int B_col = col;
        if (B_row < n && B_col < n)
        {
            Bs[ty][tx] = B[B_row * n + B_col];
        }
        else
        {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
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

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
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

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMemcpyAsync(d_a, a.data(), matrix_size, cudaMemcpyHostToDevice, stream1);

    cudaMemcpyAsync(d_b, b.data(), matrix_size, cudaMemcpyHostToDevice, stream2);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    if (n <= 256)
    {
        dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        block_gemm_kernel_small<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
    }
    else if (n <= 2048)
    {
        const int elements_per_thread = 4;
        int threads_per_dim = TILE_SIZE / elements_per_thread;

        dim3 block_dim(threads_per_dim, threads_per_dim);
        dim3 grid_dim((n + TILE_SIZE - 1) / TILE_SIZE,
                      (n + TILE_SIZE - 1) / TILE_SIZE);

        block_gemm_kernel_optimized<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
    }
    else
    {
        dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        block_gemm_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Ошибка запуска ядра CUDA: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpyAsync(c.data(), d_c, matrix_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return c;
}