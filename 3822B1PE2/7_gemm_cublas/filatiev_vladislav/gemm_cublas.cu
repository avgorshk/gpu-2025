#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("invalid size");
    }

    std::vector<float> c(n * n);
    if (n == 0) return c;

    float* d_a, * d_b, * d_c;
    size_t size = n * n * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    std::vector<float> a_colmajor(n * n);
    std::vector<float> b_colmajor(n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a_colmajor[j * n + i] = a[i * n + j];
            b_colmajor[j * n + i] = b[i * n + j];
        }
    }

    cudaMemcpy(d_a, a_colmajor.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_colmajor.data(), size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_a, n,
        d_b, n,
        &beta,
        d_c, n);

    std::vector<float> c_colmajor(n * n);
    cudaMemcpy(c_colmajor.data(), d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i * n + j] = c_colmajor[j * n + i];
        }
    }

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}