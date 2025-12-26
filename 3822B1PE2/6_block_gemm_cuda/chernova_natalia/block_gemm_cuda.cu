#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define TILE_SIZE 32
#define SUB_TILE 4

__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int n)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    float c[SUB_TILE][SUB_TILE] = {0};

    int block_row = by * TILE_SIZE;
    int block_col = bx * TILE_SIZE;

    int thread_row = ty * SUB_TILE;
    int thread_col = tx * SUB_TILE;

    __shared__ float A_s[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float B_s[TILE_SIZE][TILE_SIZE + 1];

    int num_blocks = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int k_block = 0; k_block < num_blocks; k_block++)
    {
        for (int i = 0; i < SUB_TILE; i++)
        {
            for (int j = 0; j < SUB_TILE; j++)
            {
                int local_row = thread_row + i;
                int local_col = thread_col + j;

                int a_global_row = block_row + local_row;
                int a_global_col = k_block * TILE_SIZE + local_col;
                if ((a_global_row < n) && (a_global_col < n))
                    A_s[local_row][local_col] = A[a_global_row * n + a_global_col];
                else
                    A_s[local_row][local_col] = 0.0f;
                int b_global_row = k_block * TILE_SIZE + local_row;
                int b_global_col = block_col + local_col;
                if ((b_global_row < n) && (b_global_col < n))
                    B_s[local_row][local_col] = B[b_global_row * n + b_global_col];
                else
                    B_s[local_row][local_col] = 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
        {
            for (int i = 0; i < SUB_TILE; i++)
            {
                float a_val = A_s[thread_row + i][k];
                for (int j = 0; j < SUB_TILE; j++)
                {
                    c[i][j] += a_val * B_s[k][thread_col + j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < SUB_TILE; i++)
    {
        for (int j = 0; j < SUB_TILE; j++)
        {
            int global_row = block_row + thread_row + i;
            int global_col = block_col + thread_col + j;
            if ((global_row < n) && (global_col < n))
            {
                C[global_row * n + global_col] = c[i][j];
            }
        }
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a, const std::vector<float> &b, int n)
{

    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n))
    {
        std::cerr << "Error: Matrix sizes don't match!" << std::endl;
        return std::vector<float>();
    }
    float *d_a, *d_b, *d_c;
    int num_elements = n * n;
    std::vector<float> c(num_elements, 0.0f);
    size_t byte_size = num_elements * sizeof(float);

    cudaMalloc(&d_a, byte_size);
    cudaMalloc(&d_b, byte_size);
    cudaMalloc(&d_c, byte_size);

    cudaMemcpy(d_a, a.data(), byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), byte_size, cudaMemcpyHostToDevice);

    int threads_per_side = TILE_SIZE / SUB_TILE;
    dim3 blockDim(threads_per_side, threads_per_side);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE,
                 (n + TILE_SIZE - 1) / TILE_SIZE);

    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, byte_size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}