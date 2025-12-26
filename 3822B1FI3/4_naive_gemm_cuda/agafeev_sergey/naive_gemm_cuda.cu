#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void NaiveMatrixMultiply(float* matA, float* matB, float* matC, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < dim && col < dim) {
        float sum = 0.0f;
        for (int k = 0; k < dim; ++k) {
            sum += matA[row * dim + k] * matB[k * dim + col];
        }
        matC[row * dim + col] = sum;
    } 
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& matA,
                                  const std::vector<float>& matB,
                                  int dim) {
    std::vector<float> matC(dim * dim);
    float *dev_matA, *dev_matB, *dev_matC;
    int buffer_size = dim * dim * sizeof(float);

    cudaMalloc(&dev_matA, buffer_size);
    cudaMalloc(&dev_matB, buffer_size);
    cudaMalloc(&dev_matC, buffer_size);

    cudaMemcpy(dev_matA, matA.data(), buffer_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matB, matB.data(), buffer_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((dim + 31) / 32, (dim + 31) / 32);

    NaiveMatrixMultiply<<<numBlocks, threadsPerBlock>>>(dev_matA, dev_matB, dev_matC, dim);
    cudaMemcpy(matC.data(), dev_matC, buffer_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_matB);
    cudaFree(dev_matC);
    cudaFree(dev_matA);

    
    return matC;
}