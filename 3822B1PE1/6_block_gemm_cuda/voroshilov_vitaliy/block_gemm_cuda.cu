#include "block_gemm_cuda.h"

#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void block_gemm_kernel(const float* A, const float* B, float* C, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_blocks; t++) {
        int Acol = t * BLOCK_SIZE + threadIdx.x;
        int Brow = t * BLOCK_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (Acol < n) ? A[row*n + Acol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (Brow < n) ? B[Brow*n + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            int idx = t*BLOCK_SIZE + k;
            if (idx < n)
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = n * n * sizeof(float);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(num_blocks, num_blocks);

    block_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    std::vector<float> c(n * n, 0.0f);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
