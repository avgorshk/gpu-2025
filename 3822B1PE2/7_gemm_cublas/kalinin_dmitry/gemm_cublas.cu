#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

static cublasHandle_t cublas_handle = nullptr;
static bool handle_initialized = false;

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (!handle_initialized) {
        cublasCreate(&cublas_handle);
        handle_initialized = true;
    }
    
    size_t size = n * n * sizeof(float);
    std::vector<float> c(n * n);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    std::vector<float> a_t(n * n), b_t(n * n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a_t[j * n + i] = a[i * n + j];
            b_t[j * n + i] = b[i * n + j];
        }
    }
    
    cudaMemcpy(d_a, a_t.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_t.data(), size, cudaMemcpyHostToDevice);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                &alpha, d_b, n, d_a, n, &beta, d_c, n);
    
    std::vector<float> c_t(n * n);
    cudaMemcpy(c_t.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[i * n + j] = c_t[j * n + i];
        }
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}

