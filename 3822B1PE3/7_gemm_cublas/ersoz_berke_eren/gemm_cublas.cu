#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) {
        return {};
    }

    const size_t matrixSize = static_cast<size_t>(n) * n;
    if (a.size() != matrixSize || b.size() != matrixSize) {
        throw std::invalid_argument("Matrix sizes do not match n*n");
    }

    const size_t bytes = matrixSize * sizeof(float);
    std::vector<float> c(matrixSize, 0.0f);

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;

    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);

    cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major order, but our matrices are row-major
    // C = A * B in row-major is equivalent to C^T = B^T * A^T in column-major
    // So we compute: C = B * A with swapped order
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        deviceB, n,
        deviceA, n,
        &beta,
        deviceC, n
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        cublasDestroy(handle);
        throw std::runtime_error("cublasSgemm failed");
    }

    cudaMemcpy(c.data(), deviceC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cublasDestroy(handle);

    return c;
}
