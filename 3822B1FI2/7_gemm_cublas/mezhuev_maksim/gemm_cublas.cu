#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    std::vector<float> c(static_cast<size_t>(n) * n, 0.0f);

    if (n <= 0) {
        return c;
    }

    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;

    if (cudaMalloc(&dA, bytes) != cudaSuccess ||
        cudaMalloc(&dB, bytes) != cudaSuccess ||
        cudaMalloc(&dC, bytes) != cudaSuccess) {
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return c;
    }

    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return c;
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasStatus_t stat = cublasSgemm(
        handle,
        CUBLAS_OP_N,  
        CUBLAS_OP_N,  
        n,           
        n,          
        n,           
        &alpha,
        dB, n,        
        dA, n,       
        &beta,
        dC, n         
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return c;
    }

    cudaMemcpy(c.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return c;
}
