#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cassert>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    
    assert(a.size() == static_cast<size_t>(n * n));
    assert(b.size() == static_cast<size_t>(n * n));
    
    static cublasHandle_t handle = nullptr;
    static bool initialized = false;
    if (!initialized) {
        cublasCreate(&handle);
        initialized = true;
    }
    
    std::vector<float> c(n * n, 0.0f);
    
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t matrixSize = n * n * sizeof(float);
    
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&d_a, matrixSize);
    if (cudaStatus != cudaSuccess) throw std::runtime_error("cudaMalloc failed for d_a");
    
    cudaStatus = cudaMalloc(&d_b, matrixSize);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_a);
        throw std::runtime_error("cudaMalloc failed for d_b");
    }
    
    cudaStatus = cudaMalloc(&d_c, matrixSize);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        throw std::runtime_error("cudaMalloc failed for d_c");
    }

    cudaStatus = cudaMemcpy(d_a, a.data(), matrixSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy failed for d_a");
    }
    
    cudaStatus = cudaMemcpy(d_b, b.data(), matrixSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy failed for d_b");
    }
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t cublasStatus = cublasSgemm(handle,
                                              CUBLAS_OP_T,
                                              CUBLAS_OP_T,
                                              n, n, n,
                                              &alpha,
                                              d_b, n,
                                              d_a, n,
                                              &beta,
                                              d_c, n);
    
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("cublasSgemm failed");
    }
    
    cudaStatus = cudaMemcpy(c.data(), d_c, matrixSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy failed for result");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::vector<float> result(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i * n + j] = c[j * n + i];
        }
    }
    
    return result;
}