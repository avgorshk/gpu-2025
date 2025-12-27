#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrixMulTiledKernel(const float *A, const float *B, float *C,
                                     int n)
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

std::vector<float> NaiveGemmCUDA(const std::vector<float> &A,
                                 const std::vector<float> &B,
                                 int n)
{

    float *d_A, *d_B, *d_C;
    size_t matrixSize = n * n * sizeof(float);

    cudaMalloc(&d_A, matrixSize);
    cudaMalloc(&d_B, matrixSize);
    cudaMalloc(&d_C, matrixSize);

    cudaMemcpy(d_A, A.data(), matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), matrixSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16); 
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (n + blockSize.y - 1) / blockSize.y);

    matrixMulTiledKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    std::vector<float> C(n * n);
    cudaMemcpy(C.data(), d_C, matrixSize, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}