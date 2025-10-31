#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {

    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("not n*n");
    }

    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("power of 2");
    }

    std::vector<float> result(n * n);
    if (n == 0) return result;

    size_t memory_size = n * n * sizeof(float);
    float *device_a, *device_b, *device_c;

    cudaMalloc(&device_a, memory_size);
    cudaMalloc(&device_b, memory_size);
    cudaMalloc(&device_c, memory_size);

    cudaMemcpy(device_a, a.data(), memory_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b.data(), memory_size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha_val = 1.0f;
    const float beta_val = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n,
                n,
                n,
                &alpha_val,
                device_b, n,
                device_a, n,
                &beta_val,
                device_c, n);

    cudaMemcpy(result.data(), device_c, memory_size, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return result;
}