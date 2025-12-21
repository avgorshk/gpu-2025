#include "block_gemm_cuda.h"
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


constexpr int BLOCK_SIZE = 32;

__global__ void BlockGemmKernel(
    const float *a,
    const float *b,
    float *c,
    int n
) {
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    int global_row = blockIdx.y * BLOCK_SIZE + local_row;
    int global_col = blockIdx.x * BLOCK_SIZE + local_col;

    __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;
    for (int i = 0; i < n / BLOCK_SIZE; i++) {
        tile_a[local_row][local_col] = a[global_row * n + (i * BLOCK_SIZE + local_col)];
        tile_b[local_row][local_col] = b[(i * BLOCK_SIZE + local_row) * n + global_col];
        __syncthreads();
#pragma unroll
        for (int j = 0; j < BLOCK_SIZE; j++) {
            sum += tile_a[local_row][j] * tile_b[j][local_col];
        }
        __syncthreads();
    }

    if (global_row < n && global_col < n) {
        c[global_row * n + global_col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(
    const std::vector<float> &a,
    const std::vector<float> &b,
    int n
) {
    if (n <= 0) {
        return {};
    }

    size_t size = n * n;
    size_t data_size = size * sizeof(float);
    std::vector<float> c(size);

    float *d_a = nullptr;
    cudaMalloc(&d_a, data_size);
    float *d_b = nullptr;
    cudaMalloc(&d_b, data_size);
    float *d_c = nullptr;
    cudaMalloc(&d_c, data_size);

    cudaMemcpy(d_a, a.data(), data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), data_size, cudaMemcpyHostToDevice);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (n + block.x - 1) / block.x,
        (n + block.y - 1) / block.y
    );
    BlockGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, data_size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return c;
}
