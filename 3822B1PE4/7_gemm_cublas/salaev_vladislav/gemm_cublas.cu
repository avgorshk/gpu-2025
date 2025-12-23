#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t handle = nullptr;
static float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
static size_t allocated_n = 0;

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    
    const size_t bytes = n * n * sizeof(float);
    
    if (n > allocated_n) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        allocated_n = n;
    }
    
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b, n,
                d_a, n,
                &beta,
                d_c, n);
    
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    
    return c;
}