#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    if (a.size() != nn || b.size() != nn) {
        throw std::runtime_error("GemmCUBLAS: wrong input sizes");
    }

    std::vector<float> c(nn);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaError_t cuda_status;
    cublasStatus_t blas_status;
    cublasHandle_t handle;

    cuda_status = cudaMalloc(&d_a, nn * sizeof(float));
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("cudaMalloc d_a failed");
    }

    cuda_status = cudaMalloc(&d_b, nn * sizeof(float));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_a);
        throw std::runtime_error("cudaMalloc d_b failed");
    }

    cuda_status = cudaMalloc(&d_c, nn * sizeof(float));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        throw std::runtime_error("cudaMalloc d_c failed");
    }

    cuda_status = cudaMemcpy(d_a, a.data(), nn * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy H2D a failed");
    }

    cuda_status = cudaMemcpy(d_b, b.data(), nn * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy H2D b failed");
    }

    blas_status = cublasCreate(&handle);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        throw std::runtime_error("cublasCreate failed");
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    blas_status = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n,
        n,
        n,
        &alpha,
        d_b, n,
        d_a, n,
        &beta,
        d_c, n
    );

    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        throw std::runtime_error("cublasSgemm failed");
    }

    cuda_status = cudaMemcpy(c.data(), d_c, nn * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cublasDestroy(handle);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy D2H c failed");
    }

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
