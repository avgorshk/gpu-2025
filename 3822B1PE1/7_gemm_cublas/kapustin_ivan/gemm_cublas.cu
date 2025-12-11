#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) 
{
    if (n <= 0) {
        return {};
    }

    const size_t bytes = sizeof(float) * n * n;

    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&devA), bytes);
    cudaMalloc(reinterpret_cast<void**>(&devB), bytes);
    cudaMalloc(reinterpret_cast<void**>(&devC), bytes);

    cudaMemcpy(devA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t blas;
    cublasCreate(&blas);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasStatus_t status =
        cublasSgemm(blas,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     n,       
                     n,      
                     n,       
                     &alpha,
                     devB, n, 
                     devA, n,
                     &beta,
                     devC, n);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS SGEMM error\n";
    }

    cublasDestroy(blas);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), devC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return c;
}
