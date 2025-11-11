#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {

    cublasHandle_t handle;
    cublasCreate(&handle);

    float* data_A, * data_B, * data_C;
    cudaMalloc(&data_A, n * n * sizeof(float));
    cudaMalloc(&data_B, n * n * sizeof(float));
    cudaMalloc(&data_C, n * n * sizeof(float));

    cudaMemcpy(data_A, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_B, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        n,
        n,
        &alpha,
        data_B,
        n,
        data_A,
        n,
        &beta,
        data_C,
        n
    );

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), data_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(data_A);
    cudaFree(data_B);
    cudaFree(data_C);
    cublasDestroy(handle);

    return c;
}