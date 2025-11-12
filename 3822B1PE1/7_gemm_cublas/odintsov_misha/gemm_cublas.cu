#include "gemm_cublas.h"
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    size_t len = n * n;

    std::vector<float> output(len, 0.0f);

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc((void**)&d_a, len * sizeof(float));
    cudaMalloc((void**)&d_b, len * sizeof(float));
    cudaMalloc((void**)&d_c, len * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_a, a.data(), len * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), len * sizeof(float), cudaMemcpyHostToDevice, stream);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaStreamSynchronize(stream);  
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);
    cudaMemcpyAsync(output.data(), d_c, len * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    return output;
}
