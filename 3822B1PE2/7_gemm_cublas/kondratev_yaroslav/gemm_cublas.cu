#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n == 0) return {};

    size_t matrixSize = n * n;
    std::vector<float> c(matrixSize, 0.0f);
    size_t bytes = matrixSize * sizeof(float);    
    float *d_a{}, *d_b{}, *d_c{};
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, &alpha, d_b, n,
                d_a, n, &beta, d_c, n);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    cublasDestroy(handle);
    return c;
}