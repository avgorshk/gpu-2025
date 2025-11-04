#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    const size_t num_elements = static_cast<size_t>(n) * n;
    const size_t size_bytes = num_elements * sizeof(float);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaMalloc(&d_a, size_bytes);
    cudaMalloc(&d_b, size_bytes);
    cudaMalloc(&d_c, size_bytes);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMemcpy(d_a, a.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size_bytes, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n,
                n,
                n,
                &alpha,
                d_b,
                n,
                d_a,
                n,
                &beta,
                d_c,
                n);

    std::vector<float> c(num_elements);
    cudaMemcpy(c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost);
    
    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}