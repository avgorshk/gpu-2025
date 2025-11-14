#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <string>

inline void cudaCheck(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

inline void cublasCheck(cublasStatus_t st, const char* msg)
{
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS: ") + msg);
    }
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
    const std::vector<float>& b,
    int n)
{
    if (n <= 0) {
        return {};
    }

    const std::size_t expected = static_cast<std::size_t>(n) * n;
    if (a.size() != expected || b.size() != expected) {
        throw std::runtime_error("GemmCUBLAS: wrong matrix size");
    }

    const std::size_t bytes = expected * sizeof(float);

    float* dA = nullptr, * dB = nullptr, * dC = nullptr;

    cudaCheck(cudaMalloc(&dA, bytes), "cudaMalloc dA failed");
    cudaCheck(cudaMalloc(&dB, bytes), "cudaMalloc dB failed");
    cudaCheck(cudaMalloc(&dC, bytes), "cudaMalloc dC failed");

    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream), "cudaStreamCreate failed");

    cudaCheck(cudaMemcpyAsync(dA, a.data(), bytes,
        cudaMemcpyHostToDevice, stream),
        "Memcpy A H2D failed");
    cudaCheck(cudaMemcpyAsync(dB, b.data(), bytes,
        cudaMemcpyHostToDevice, stream),
        "Memcpy B H2D failed");

    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle), "create handle");
    cublasCheck(cublasSetStream(handle, stream), "set stream");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasCheck(
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n,
            n,
            n,
            &alpha,
            dB, n,
            dA, n,
            &beta,
            dC, n),
        "sgemm failed");

    std::vector<float> c(expected);
    cudaCheck(cudaMemcpyAsync(c.data(), dC, bytes,
        cudaMemcpyDeviceToHost, stream),
        "Memcpy C D2H failed");

    cudaCheck(cudaStreamSynchronize(stream), "stream sync failed");

    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return c;
}
