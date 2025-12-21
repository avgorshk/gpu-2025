#include "gemm_cublas.h"
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(
    const std::vector<float> &a,
    const std::vector<float> &b,
    int n
) {
    if (n <= 0) {
        return {};
    }

    size_t size = n * n;
    size_t data_size = size * sizeof(float);
    std::vector<float> c(size);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float *d_a = nullptr;
    cudaMalloc(&d_a, data_size);
    float *d_b = nullptr;
    cudaMalloc(&d_b, data_size);
    float *d_c = nullptr;
    cudaMalloc(&d_c, data_size);

    cudaMemcpy(d_a, a.data(), data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), data_size, cudaMemcpyHostToDevice);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_b, n,
        d_a, n,
        &beta,
        d_c, n
    );
    cudaMemcpy(c.data(), d_c, data_size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
    return c;
}
