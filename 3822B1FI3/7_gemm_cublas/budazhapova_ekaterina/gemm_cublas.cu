#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (a.size() != static_cast<size_t>(n * n) || b.size() != static_cast<size_t>(n * n)) {
        std::cerr << "Error: Invalid matrix sizes" << std::endl;
        return std::vector<float>();
    }
    
    std::vector<float> c(n * n);
    size_t size = n * n * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 
                n, n, n, 
                &alpha, 
                d_A, n,  
                d_B, n,  
                &beta, 
                d_C, n); 
   
    cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost);
    
    std::vector<float> result(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] = c[j * n + i]; 
        }
    }
    
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return result;
}