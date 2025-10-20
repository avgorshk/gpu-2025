#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

void checkCublasError(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error: " << msg << " - " << status << std::endl;
        throw std::runtime_error("cuBLAS error");
    }
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("Matrix sizes do not match n*n");
    }

    size_t bytes = n * n * sizeof(float);
    std::vector<float> c(n * n);

    if (n == 0) return c;

    float* d_a = nullptr, * d_b = nullptr, * d_c = nullptr;

    checkCudaError(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    checkCudaError(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    checkCudaError(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "cublasCreate");

    checkCudaError(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "Copy a to device");
    checkCudaError(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "Copy b to device");
    const float alpha = 1.0f;
    const float beta = 0.0f;

    checkCublasError(
        cublasSgemm(handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            n,
            n,
            &alpha,
            d_b, n,
            d_a, n,
            &beta,
            d_c, n),
        "cublasSgemm"
    );

    checkCudaError(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "Copy c from device");

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}