#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n == 0) return {};

    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS initialization failed");
    }

    size_t bytes = n * n * sizeof(float);
    float* d_a = nullptr, * d_b = nullptr, * d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);


    const float alpha = 1.0f;
    const float beta = 0.0f;

    stat = cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_b, n,
        d_a, n,                    
        &beta,
        d_c, n);                   
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        throw std::runtime_error("cuBLAS GEMM failed");
    }

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);

    return c;
}