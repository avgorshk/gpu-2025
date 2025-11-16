#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    size_t bytes = n * n * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                dB, n,  // B (в row-major это A)
                dA, n,  // A (в row-major это B)
                &beta,
                dC, n);
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), dC, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);
    return c;
}