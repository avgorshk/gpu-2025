#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("Matrix dimensions mismatch");
    }

    size_t bytes = n * n * sizeof(float);
    std::vector<float> c(n * n, 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        n,
        n,
        n,
        &alpha,
        d_B, n,
        d_A, n,
        &beta,
        d_C, n
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        throw std::runtime_error("cublasSgemm failed");
    }

    cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return c;
}
