#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& matrix_a,
                              const std::vector<float>& matrix_b,
                              int matrix_size) {
    std::vector<float> result(matrix_size * matrix_size);
    size_t matrix_bytes = matrix_size * matrix_size * sizeof(float);

    float* device_a;
    float* device_b;
    float* device_c;
    float* device_c_transposed;

    cudaMalloc(&device_a, matrix_bytes);
    cudaMalloc(&device_b, matrix_bytes);
    cudaMalloc(&device_c, matrix_bytes);
    cudaMalloc(&device_c_transposed, matrix_bytes);

    cublasSetMatrix(matrix_size, matrix_size, sizeof(float), matrix_a.data(), matrix_size, device_a, matrix_size);
    cublasSetMatrix(matrix_size, matrix_size, sizeof(float), matrix_b.data(), matrix_size, device_b, matrix_size);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, matrix_size, matrix_size, matrix_size,
                &alpha, device_a, matrix_size, device_b, matrix_size, &beta, device_c, matrix_size);

    cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, matrix_size, matrix_size,
                &alpha, device_c, matrix_size, &beta, nullptr, matrix_size, device_c_transposed, matrix_size);

    cublasGetMatrix(matrix_size, matrix_size, sizeof(float), device_c_transposed, matrix_size, result.data(), matrix_size);

    cudaFree(device_b);
    cudaFree(device_c);
    cudaFree(device_a);
    cudaFree(device_c_transposed);

    cublasDestroy(cublas_handle);

    return result;
}