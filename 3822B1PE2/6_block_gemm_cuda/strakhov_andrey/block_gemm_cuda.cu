#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 32
#define SUB_TILE 4

__global__ void kernel(const float *A, const float *B, float *C, int n)
{
    float res[SUB_TILE][SUB_TILE] = {0.0};
    int block_x = blockIdx.x * TILE_SIZE;
    int block_y = blockIdx.y * TILE_SIZE;
    int thread_x = threadIdx.x * SUB_TILE;
    int thread_y = threadIdx.y * SUB_TILE;

    __shared__ float sharelocal_a[TILE_SIZE][TILE_SIZE];
    __shared__ float sharelocal_b[TILE_SIZE][TILE_SIZE];

    for (int strider = 0; strider < (n + TILE_SIZE - 1) / TILE_SIZE; strider++)
    {
        for (int i = 0; i < SUB_TILE; i++)
        {
            for (int j = 0; j < SUB_TILE; j++)
            {
                int local_y = thread_y + i;
                int local_x = thread_x + j;

                int a_global_y = block_y + local_y;
                int a_global_x = strider * TILE_SIZE + local_x;
                if ((a_global_y < n) && (a_global_x < n))
                {
                    sharelocal_a[local_y][local_x] = A[a_global_y * n + a_global_x];
                }
                else
                {
                    sharelocal_a[local_y][local_x] = 0.0f;
                }
                int b_global_y = strider * TILE_SIZE + local_y;
                int b_global_x = block_x + local_x;
                if ((b_global_y < n) && (b_global_x < n))
                {
                    sharelocal_b[local_y][local_x] = B[b_global_y * n + b_global_x];
                }
                else
                {
                    sharelocal_b[local_y][local_x] = 0.0f;
                }
            }
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
        {
            for (int i = 0; i < SUB_TILE; i++)
            {
                for (int j = 0; j < SUB_TILE; j++)
                {
                    res[i][j] += sharelocal_a[thread_y + i][k] * sharelocal_b[k][thread_x + j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < SUB_TILE; i++)
    {
        for (int j = 0; j < SUB_TILE; j++)
        {
            int global_y = block_y + thread_y + i;
            int global_x = block_x + thread_x + j;
            if ((global_y < n) && (global_x < n))
            {
                C[global_y * n + global_x] = res[i][j];
            }
        }
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a, const std::vector<float> &b, int n)
{
    float *local_a, *local_b, *local_c;
    std::vector<float> res(n * n, 0.0f);
    size_t size = n * n * sizeof(float);

    cudaMalloc(&local_a, size);
    cudaMalloc(&local_b, size);
    cudaMalloc(&local_c, size);

    cudaMemcpy(local_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(local_b, b.data(), size, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE / SUB_TILE, TILE_SIZE / SUB_TILE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    kernel<<<grid, block>>>(local_a, local_b, local_c, n);

    cudaDeviceSynchronize();
    cudaMemcpy(res.data(), local_c, size, cudaMemcpyDeviceToHost);
    cudaFree(local_a);
    cudaFree(local_b);
    cudaFree(local_c);

    return res;
}