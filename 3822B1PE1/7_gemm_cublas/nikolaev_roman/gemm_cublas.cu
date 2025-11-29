#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                             const std::vector<float>& b,
                             int n) {
    assert(a.size() == n * n && b.size() == n * n);
    assert((n & (n - 1)) == 0 && "Matrix size must be power of 2");
    
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS handle creation failed");
    }
    
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    
    cudaError_t cudaErr;
    cudaErr = cudaMalloc(&d_A, size);
    if (cudaErr != cudaSuccess) {
        cublasDestroy(handle);
        throw std::runtime_error("cudaMalloc failed for d_A");
    }
    
    cudaErr = cudaMalloc(&d_B, size);
    if (cudaErr != cudaSuccess) {
        cudaFree(d_A);
        cublasDestroy(handle);
        throw std::runtime_error("cudaMalloc failed for d_B");
    }
    
    cudaErr = cudaMalloc(&d_C, size);
    if (cudaErr != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cublasDestroy(handle);
        throw std::runtime_error("cudaMalloc failed for d_C");
    }
    
    std::vector<float> a_transposed(n * n);
    std::vector<float> b_transposed(n * n);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a_transposed[j * n + i] = a[i * n + j];
            b_transposed[j * n + i] = b[i * n + j];
        }
    }
    
    cudaErr = cudaMemcpyAsync(d_A, a_transposed.data(), size, cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        throw std::runtime_error("cudaMemcpyAsync failed for d_A");
    }
    
    cudaErr = cudaMemcpyAsync(d_B, b_transposed.data(), size, cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        throw std::runtime_error("cudaMemcpyAsync failed for d_B");
    }
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    status = cublasSgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N, 
                        n,            
                        n,            
                        n,           
                        &alpha,       
                        d_A,          
                        n,            
                        d_B,          
                        n,            
                        &beta,        
                        d_C,          
                        n);           
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        throw std::runtime_error("cuBLAS Sgemm failed");
    }
    
    cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        throw std::runtime_error("cudaDeviceSynchronize failed");
    }
    
    std::vector<float> c_column_major(n * n);
    cudaErr = cudaMemcpy(c_column_major.data(), d_C, size, cudaMemcpyDeviceToHost);
    if (cudaErr != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        throw std::runtime_error("cudaMemcpy failed for result");
    }
    
    std::vector<float> c(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[i * n + j] = c_column_major[j * n + i];
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    
    return c;
}