#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_DIM 32

namespace {
    __global__ void MatrixMultiplyKernel(
        const float* mat_A,
        const float* mat_B,
        float* mat_C,
        int n) {

        __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
        __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

        int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
        int col = blockIdx.x * BLOCK_DIM + threadIdx.x;

        float acc = 0.0f;

        for (int tile = 0; tile < n; tile += BLOCK_DIM) {
            bool loadA = (row < n) && (tile + threadIdx.x < n);
            bool loadB = (col < n) && (tile + threadIdx.y < n);

            shared_A[threadIdx.y][threadIdx.x] =
                loadA ? mat_A[row * n + tile + threadIdx.x] : 0.0f;

            shared_B[threadIdx.y][threadIdx.x] =
                loadB ? mat_B[(tile + threadIdx.y) * n + col] : 0.0f;

            __syncthreads();

            #pragma unroll
            for (int k = 0; k < BLOCK_DIM; ++k) {
                acc += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
            }

            __syncthreads();
        }

        if (row < n && col < n) {
            mat_C[row * n + col] = acc;
        }
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    float *dev_A = nullptr, *dev_B = nullptr, *dev_C = nullptr;

    cudaMalloc(&dev_A, bytes);
    cudaMalloc(&dev_B, bytes);
    cudaMalloc(&dev_C, bytes);

    cudaMemcpy(dev_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((n + BLOCK_DIM - 1) / BLOCK_DIM,
              (n + BLOCK_DIM - 1) / BLOCK_DIM);

    MatrixMultiplyKernel<<<grid, block>>>(dev_A, dev_B, dev_C, n);
    cudaDeviceSynchronize();

    std::vector<float> c(static_cast<size_t>(n) * n);
    cudaMemcpy(c.data(), dev_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return c;
}
