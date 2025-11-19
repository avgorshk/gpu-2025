#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0 || a.size() != static_cast<size_t>(n * n) || 
        b.size() != static_cast<size_t>(n * n)) {
        return {};
    }

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return {};
    }

    size_t mat_bytes = static_cast<size_t>(n) * n * sizeof(float);
    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;

    cudaError_t cuda_ret = cudaMalloc(&dev_a, mat_bytes);
    if (cuda_ret != cudaSuccess) {
        cublasDestroy(handle);
        return {};
    }
    
    cuda_ret = cudaMalloc(&dev_b, mat_bytes);
    if (cuda_ret != cudaSuccess) {
        cudaFree(dev_a);
        cublasDestroy(handle);
        return {};
    }
    
    cuda_ret = cudaMalloc(&dev_c, mat_bytes);
    if (cuda_ret != cudaSuccess) {
        cudaFree(dev_a);
        cudaFree(dev_b);
        cublasDestroy(handle);
        return {};
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cublasDestroy(handle);
        return {};
    }

    cublasSetStream(handle, stream);

    cuda_ret = cudaMemcpyAsync(dev_a, a.data(), mat_bytes, cudaMemcpyHostToDevice, stream);
    if (cuda_ret != cudaSuccess) {
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cudaStreamDestroy(stream);
        cublasDestroy(handle);
        return {};
    }

    cuda_ret = cudaMemcpyAsync(dev_b, b.data(), mat_bytes, cudaMemcpyHostToDevice, stream);
    if (cuda_ret != cudaSuccess) {
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cudaStreamDestroy(stream);
        cublasDestroy(handle);
        return {};
    }

    float one = 1.0f;
    float zero = 0.0f;

    cublasStatus_t stat = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, n, n,
        &one,
        dev_b, n,
        dev_a, n,
        &zero,
        dev_c, n
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cudaStreamDestroy(stream);
        cublasDestroy(handle);
        return {};
    }

    std::vector<float> result(static_cast<size_t>(n) * n);
    cuda_ret = cudaMemcpyAsync(result.data(), dev_c, mat_bytes, cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cublasDestroy(handle);

    if (cuda_ret != cudaSuccess) {
        return {};
    }

    return result;
}
