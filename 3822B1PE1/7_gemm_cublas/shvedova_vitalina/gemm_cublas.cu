#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    assert(a.size() == static_cast<size_t>(n * n));
    assert(b.size() == static_cast<size_t>(n * n));

    const size_t bytes = n * n * sizeof(float);

    float *devA, *devB, *devC;
    cudaMalloc(&devA, bytes);
    cudaMalloc(&devB, bytes);
    cudaMalloc(&devC, bytes);

    cudaMemcpyAsync(devA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(devB, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                devB, n,
                devA, n,
                &beta,
                devC, n);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), devC, bytes, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return c;
}
