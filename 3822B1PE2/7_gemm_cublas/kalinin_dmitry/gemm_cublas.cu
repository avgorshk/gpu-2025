#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t bytes = n * n * sizeof(float);
    std::vector<float> result(n * n);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaStream_t compute_stream;
    cudaStreamCreate(&compute_stream);
    cublasSetStream(cublas_handle, compute_stream);

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, compute_stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, compute_stream);

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                &alpha, d_b, n, d_a, n, &beta, d_c, n);

    cudaMemcpyAsync(result.data(), d_c, bytes, cudaMemcpyDeviceToHost, compute_stream);
    cudaStreamSynchronize(compute_stream);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaStreamDestroy(compute_stream);
    cublasDestroy(cublas_handle);

    return result;
}

