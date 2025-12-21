#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    std::vector<float> result(n * n);
    
    float *deviceA, *deviceB, *deviceResult;
    size_t bytes = n * n * sizeof(float);
    
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceResult, bytes);
    
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    
    cublasSetMatrix(n, n, sizeof(float), a.data(), n, deviceA, n);
    cublasSetMatrix(n, n, sizeof(float), b.data(), n, deviceB, n);
    
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    cublasSgemm(cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                deviceB, n,
                deviceA, n,
                &beta,
                deviceResult, n);
    
    cublasGetMatrix(n, n, sizeof(float), deviceResult, n, result.data(), n);
    
    cublasDestroy(cublasHandle);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceResult);
    
    return result;
}
