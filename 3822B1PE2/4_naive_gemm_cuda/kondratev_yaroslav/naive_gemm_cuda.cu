#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void naiveGemmKernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t matrixSize = n * n;
    std::vector<float> c(matrixSize, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaMalloc(&d_a, matrixSize * sizeof(float));
    cudaMalloc(&d_b, matrixSize * sizeof(float));
    cudaMalloc(&d_c, matrixSize * sizeof(float));

    cudaMemcpy(d_a, a.data(), matrixSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), matrixSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16); 
    dim3 numBlocks((n + blockSize.x - 1) / blockSize.x,
                   (n + blockSize.y - 1) / blockSize.y);

    naiveGemmKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}