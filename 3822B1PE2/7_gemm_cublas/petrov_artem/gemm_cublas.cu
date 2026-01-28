#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t stat = call; \
        if (stat != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error: " << stat << std::endl; \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while(0)

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b,int n) {
    if (n == 0) return {};
    
    if (a.size() != static_cast<size_t>(n * n) || 
        b.size() != static_cast<size_t>(n * n)) {
        throw std::runtime_error("Matrix size mismatch");
    }
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    size_t bytes = n * n * sizeof(float);
    
    float* pinned_c = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned_c, bytes));
    
    cudaStream_t compute_stream, copy_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&copy_stream));
    
    CUBLAS_CHECK(cublasSetStream(handle, compute_stream));
    
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    CUDA_CHECK(cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, copy_stream));
    
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    CUBLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_T, CUBLAS_OP_T,n, n, n,&alpha,d_a, n, d_b, n,&beta,d_c, n));
    
    CUDA_CHECK(cudaMemcpyAsync(pinned_c, d_c, bytes, cudaMemcpyDeviceToHost, compute_stream));
    
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    
    std::vector<float> c(n * n);
    std::copy(pinned_c, pinned_c + n * n, c.begin());
    
    CUDA_CHECK(cudaFreeHost(pinned_c));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    
    CUBLAS_CHECK(cublasDestroy(handle));
    
    return c;
}