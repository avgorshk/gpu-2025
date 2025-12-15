#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void gemm_kernel(const float* A, const float* B, float* C, int n) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;

#pragma unroll 4
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {

    const int size = n * n;
    float* data_A, * data_B, * data_C;
    cudaMalloc(&data_A, size * sizeof(float));
    cudaMalloc(&data_B, size * sizeof(float));
    cudaMalloc(&data_C, size * sizeof(float));

    cudaMemcpy(data_A, a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_B, b.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    gemm_kernel<<<blocks, threads>>>(data_A, data_B, data_C, n);
    cudaDeviceSynchronize();

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), data_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(data_A);
    cudaFree(data_B);
    cudaFree(data_C);

    return c;

}
