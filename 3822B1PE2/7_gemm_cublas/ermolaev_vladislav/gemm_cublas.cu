#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    std::vector<float> c(n * n);
    
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cublasSetMatrix(n, n, sizeof(float), a.data(), n, d_A, n);
    cublasSetMatrix(n, n, sizeof(float), b.data(), n, d_B, n);
    
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_B, n,
                d_A, n,
                &beta,
                d_C, n);
    
    cublasGetMatrix(n, n, sizeof(float), d_C, n, c.data(), n);
    
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return c;
}
