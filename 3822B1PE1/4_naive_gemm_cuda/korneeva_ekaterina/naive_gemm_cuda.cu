#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void unrolledGemmKernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        
        int k = 0;
        for (; k <= n - 4; k += 4) {
            sum += A[row * n + k] * B[k * n + col] +
                   A[row * n + k + 1] * B[(k + 1) * n + col] +
                   A[row * n + k + 2] * B[(k + 2) * n + col] +
                   A[row * n + k + 3] * B[(k + 3) * n + col];
        }
        
        for (; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        
        C[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (a.size() != n * n || b.size() != n * n) {
        std::cerr << "Error: Matrix size mismatch!" << std::endl;
        return std::vector<float>();
    }
    
    std::vector<float> c(n * n);
    
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t size = n * n * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    CUDA_CHECK(cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice));
    
    dim3 blockSize(32, 8);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, 
                  (n + blockSize.y - 1) / blockSize.y);
    
    unrolledGemmKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return c;
}