#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t num_bytes = static_cast<size_t>(n) * n * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    if (cudaMalloc(&dA, num_bytes) != cudaSuccess ||
        cudaMalloc(&dB, num_bytes) != cudaSuccess ||
        cudaMalloc(&dC, num_bytes) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory");
    }

    if (cudaMemcpy(dA, a.data(), num_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dB, b.data(), num_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("Failed to copy data to device");
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                dB, n,
                dA, n,
                &beta,
                dC, n);

    std::vector<float> c(n * n, 0.0f);
    cudaMemcpy(c.data(), dC, num_bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);

    return c;
}
