#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    std::vector<float> c(n * n);
    const size_t bytes = n * n * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cublasSetMatrix(n, n, sizeof(float), a.data(), n, d_a, n);
    cublasSetMatrix(n, n, sizeof(float), b.data(), n, d_b, n);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_b, n, d_a, n, &beta, d_c, n);

    cublasGetMatrix(n, n, sizeof(float), d_c, n, c.data(), n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cublasDestroy(handle);
    return c;
}
