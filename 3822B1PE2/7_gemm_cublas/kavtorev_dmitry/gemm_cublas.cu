#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t size = n * n * sizeof(float);
    std::vector<float> c(n * n);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    std::vector<float> a_transposed(n * n);
    std::vector<float> b_transposed(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a_transposed[j * n + i] = a[i * n + j];
            b_transposed[j * n + i] = b[i * n + j];
        }
    }
    
    cudaMemcpy(d_a, a_transposed.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_transposed.data(), size, cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                &alpha, d_b, n, d_a, n, &beta, d_c, n);
    
    cudaDeviceSynchronize();
    
    std::vector<float> c_transposed(n * n);
    cudaMemcpy(c_transposed.data(), d_c, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[i * n + j] = c_transposed[j * n + i];
        }
    }
    
    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}

