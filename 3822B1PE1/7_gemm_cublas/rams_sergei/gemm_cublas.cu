#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *_a, *_b, *_c;
    size_t size = n * n * sizeof(float);

    cudaMalloc(&_a, size);
    cudaMalloc(&_b, size);
    cudaMalloc(&_c, size);
    cudaMemcpy(_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b.data(), size, cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, _b, n, _a, n, &beta, _c, n);

    std::vector<float> result(n * n);
    cudaMemcpy(result.data(), _c, size, cudaMemcpyDeviceToHost);

    cudaFree(_a);
    cudaFree(_b);
    cudaFree(_c);
    cublasDestroy(handle);

    return result;
}
