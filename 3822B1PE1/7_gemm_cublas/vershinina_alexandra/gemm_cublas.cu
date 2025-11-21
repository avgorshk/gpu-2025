#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <string>

static inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + msg + " : " + cudaGetErrorString(e));
    }
}
static inline void checkCublas(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error: ") + msg);
    }
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) return {};
    const size_t N = static_cast<size_t>(n);
    const size_t elems = N * N;
    if (a.size() != elems || b.size() != elems) {
        throw std::invalid_argument("Input matrices must have size n*n");
    }

    const size_t bytes = elems * sizeof(float);

    float *hA = nullptr, *hB = nullptr, *hC = nullptr;
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;

    try {
        checkCuda(cudaMallocHost((void**)&hA, bytes), "cudaMallocHost hA");
        checkCuda(cudaMallocHost((void**)&hB, bytes), "cudaMallocHost hB");
        checkCuda(cudaMallocHost((void**)&hC, bytes), "cudaMallocHost hC");

        std::memcpy(hA, a.data(), bytes);
        std::memcpy(hB, b.data(), bytes);

        checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        checkCublas(cublasCreate(&handle), "cublasCreate");
        checkCublas(cublasSetStream(handle, stream), "cublasSetStream");

        checkCuda(cudaMalloc((void**)&dA, bytes), "cudaMalloc dA");
        checkCuda(cudaMalloc((void**)&dB, bytes), "cudaMalloc dB");
        checkCuda(cudaMalloc((void**)&dC, bytes), "cudaMalloc dC");

        checkCuda(cudaMemcpyAsync(dA, hA, bytes, cudaMemcpyHostToDevice, stream), "H2D dA");
        checkCuda(cudaMemcpyAsync(dB, hB, bytes, cudaMemcpyHostToDevice, stream), "H2D dB");

        const float alpha = 1.0f;
        const float beta  = 0.0f;

        checkCublas(
            cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, n, n,
                        &alpha,
                        dB, n,    
                        dA, n,    
                        &beta,
                        dC, n),
            "cublasSgemm"
        );

        checkCuda(cudaMemcpyAsync(hC, dC, bytes, cudaMemcpyDeviceToHost, stream), "D2H dC");
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        std::vector<float> result(elems);
        std::memcpy(result.data(), hC, bytes);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
        cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);

        return result;
    } catch (...) {
        if (dA) cudaFree(dA);
        if (dB) cudaFree(dB);
        if (dC) cudaFree(dC);
        if (handle) cublasDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
        if (hA) cudaFreeHost(hA);
        if (hB) cudaFreeHost(hB);
        if (hC) cudaFreeHost(hC);
        throw;
    }
}
