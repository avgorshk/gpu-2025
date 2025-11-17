#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
    const std::vector<float>& b,
    int n)
{
    if ((int)a.size() != n * n || (int)b.size() != n * n)
        throw std::runtime_error("Matrix size mismatch");

    std::vector<float> c(n * n);

    float* dA = nullptr, * dB = nullptr, * dC = nullptr;

    cudaMalloc((void**)&dA, n * n * sizeof(float));
    cudaMalloc((void**)&dB, n * n * sizeof(float));
    cudaMalloc((void**)&dC, n * n * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(dA, a.data(), n * n * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dB, b.data(), n * n * sizeof(float),
        cudaMemcpyHostToDevice, stream);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        n,
        n,
        n,
        &alpha,
        dB, n,
        dA, n,
        &beta,
        dC, n
    );

    cudaMemcpyAsync(c.data(), dC, n * n * sizeof(float),
        cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return c;
}
