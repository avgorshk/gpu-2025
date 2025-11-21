#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>

void checkCudaError(cudaError_t error_code, const char* error_message) {
    if (error_code != cudaSuccess) {
        std::cerr << "CUDA Error (" << error_message << "): "
            << cudaGetErrorString(error_code) << std::endl;
        throw std::runtime_error(cudaGetErrorString(error_code));
    }
}

void checkCublasError(cublasStatus_t status, const char* operation_name) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error (" << operation_name << "): " << status << std::endl;
        throw std::runtime_error("cuBLAS operation failed");
    }
}

std::vector<float> transposeMatrix(const std::vector<float>& matrix, int n) {
    std::vector<float> transposed(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            transposed[j * n + i] = matrix[i * n + j];
        }
    }
    return transposed;
}

std::vector<float> GemmCUBLAS(const std::vector<float>& matrix_a,
    const std::vector<float>& matrix_b,
    int matrix_size) {
    if (matrix_a.size() != static_cast<size_t>(matrix_size * matrix_size) ||
        matrix_b.size() != static_cast<size_t>(matrix_size * matrix_size)) {
        throw std::invalid_argument("Matrix dimensions do not match specified size");
    }

    std::vector<float> result_matrix(matrix_size * matrix_size, 0.0f);
    if (matrix_size == 0) {
        return result_matrix;
    }

    std::vector<float> a_transposed = transposeMatrix(matrix_a, matrix_size);
    std::vector<float> b_transposed = transposeMatrix(matrix_b, matrix_size);

    size_t memory_bytes = matrix_size * matrix_size * sizeof(float);
    float* device_a = nullptr, * device_b = nullptr, * device_c = nullptr;

    checkCudaError(cudaMalloc(&device_a, memory_bytes), "Allocating device memory for matrix A");
    checkCudaError(cudaMalloc(&device_b, memory_bytes), "Allocating device memory for matrix B");
    checkCudaError(cudaMalloc(&device_c, memory_bytes), "Allocating device memory for result");

    cublasHandle_t cublas_handle;
    checkCublasError(cublasCreate(&cublas_handle), "Creating cuBLAS handle");

    cudaStream_t computation_stream;
    checkCudaError(cudaStreamCreate(&computation_stream), "Creating CUDA stream");
    checkCublasError(cublasSetStream(cublas_handle, computation_stream), "Setting cuBLAS stream");

    checkCudaError(cudaMemcpyAsync(device_a, a_transposed.data(), memory_bytes,
        cudaMemcpyHostToDevice, computation_stream),
        "Copying matrix A to device");
    checkCudaError(cudaMemcpyAsync(device_b, b_transposed.data(), memory_bytes,
        cudaMemcpyHostToDevice, computation_stream),
        "Copying matrix B to device");

    const float alpha_value = 1.0f;
    const float beta_value = 0.0f;

    checkCublasError(cublasSgemm(cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        matrix_size,
        matrix_size,
        matrix_size,
        &alpha_value,
        device_a,
        matrix_size,
        device_b,
        matrix_size,
        &beta_value,
        device_c,
        matrix_size),
        "Executing cuBLAS SGEMM");
    std::vector<float> result_colmajor(matrix_size * matrix_size);
    checkCudaError(cudaMemcpyAsync(result_colmajor.data(), device_c, memory_bytes,
        cudaMemcpyDeviceToHost, computation_stream),
        "Copying result from device");

    checkCudaError(cudaStreamSynchronize(computation_stream), "Synchronizing stream");

    result_matrix = transposeMatrix(result_colmajor, matrix_size);

    checkCublasError(cublasDestroy(cublas_handle), "Destroying cuBLAS handle");
    checkCudaError(cudaStreamDestroy(computation_stream), "Destroying CUDA stream");

    checkCudaError(cudaFree(device_a), "Freeing device memory for A");
    checkCudaError(cudaFree(device_b), "Freeing device memory for B");
    checkCudaError(cudaFree(device_c), "Freeing device memory for result");

    return result_matrix;
}