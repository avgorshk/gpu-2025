#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    std::vector<float> c(n * n);
    
    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static int allocated_size = 0;
    static cublasHandle_t handle = nullptr;
    
    int size = n * n * sizeof(float);
    
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    
    if (allocated_size < size) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);
        allocated_size = size;
    }
    
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // C = A * B (row-major) equals C^T = B^T * A^T (column-major)
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b, n,
                d_a, n,
                &beta,
                d_c, n);
    
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    return c;
}
