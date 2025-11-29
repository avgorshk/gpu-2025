#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <mutex>

static float* dA = nullptr;
static float* dB = nullptr;
static float* dC = nullptr;

static float* hA_pinned = nullptr;
static float* hB_pinned = nullptr;
static float* hC_pinned = nullptr;

static int allocatedN = 0;

static cudaStream_t stream = nullptr;
static cublasHandle_t handle = nullptr;

static std::mutex initMutex;

static void initIfNeeded(int n)
{
    std::lock_guard<std::mutex> lock(initMutex);

    if (allocatedN == n)
        return;

    if (allocatedN != 0) {
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        cudaFreeHost(hA_pinned);
        cudaFreeHost(hB_pinned);
        cudaFreeHost(hC_pinned);

        cudaStreamDestroy(stream);
        cublasDestroy(handle);
    }

    size_t bytes = n * n * sizeof(float);

    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMallocHost(&hA_pinned, bytes);
    cudaMallocHost(&hB_pinned, bytes);
    cudaMallocHost(&hC_pinned, bytes);

    cudaStreamCreate(&stream);

    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    
    allocatedN = n;
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    if ((int)a.size() != n * n || (int)b.size() != n * n)
        throw std::runtime_error("Matrix size mismatch");

    initIfNeeded(n);

    size_t bytes = n * n * sizeof(float);

    memcpy(hA_pinned, a.data(), bytes);
    memcpy(hB_pinned, b.data(), bytes);

    cudaMemcpyAsync(dA, hA_pinned, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dB, hB_pinned, bytes, cudaMemcpyHostToDevice, stream);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        n, n, n,
        &alpha,
        dA, n,
        dB, n,
        &beta,
        dC, n
    );

    cudaMemcpyAsync(hC_pinned, dC, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    std::vector<float> c(n * n);
    memcpy(c.data(), hC_pinned, bytes);

    return c;
}
