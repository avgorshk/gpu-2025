#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t size = n * n * sizeof(float);
    std::vector<float> c(n * n);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_T,
                n, n, n,
                &alpha,
                d_a, n,
                d_b, n,
                &beta,
                d_c, n);
    
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
    
    return c;
}