#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("Matrix size mismatch");
    }

    size_t matrix_size = n * n * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, matrix_size);
    cudaMalloc(&d_b, matrix_size);
    cudaMalloc(&d_c, matrix_size);

    std::vector<float> c(n * n);
    std::vector<float> a_colmajor(n * n);
    std::vector<float> b_colmajor(n * n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a_colmajor[j * n + i] = a[i * n + j];
            b_colmajor[j * n + i] = b[i * n + j];
        }
    }

    cudaMemcpy(d_a, a_colmajor.data(), matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_colmajor.data(), matrix_size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_a, n,
                d_b, n,
                &beta,
                d_c, n);

    cudaMemcpy(c.data(), d_c, matrix_size, cudaMemcpyDeviceToHost);

    std::vector<float> result(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i * n + j] = c[j * n + i];
        }
    }

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}