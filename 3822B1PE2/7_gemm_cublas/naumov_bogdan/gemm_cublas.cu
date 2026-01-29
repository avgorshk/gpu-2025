#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>

static cublasHandle_t g_cublas_handle = nullptr;
static std::once_flag g_cublas_init_flag;

struct DeviceMemoryPool {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    size_t current_size = 0;
    
    void ensure_size(size_t n, size_t size) {
        if (current_size < size) {
            free();
            cudaMalloc(&d_a, size);
            cudaMalloc(&d_b, size);
            cudaMalloc(&d_c, size);
            current_size = size;
        }
    }
    
    void free() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        d_a = d_b = d_c = nullptr;
        current_size = 0;
    }
    
    ~DeviceMemoryPool() { free(); }
};

static DeviceMemoryPool g_mem_pool;

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (a.size() != static_cast<size_t>(n * n) || 
        b.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("Invalid matrix size");
    }
    
    std::call_once(g_cublas_init_flag, []() {
        cublasCreate(&g_cublas_handle);
        cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH);
    });
    
    std::vector<float> c(n * n);
    
    size_t matrix_size = n * n * sizeof(float);
    g_mem_pool.ensure_size(n, matrix_size);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(g_cublas_handle, stream);
    
    cudaMemcpyAsync(g_mem_pool.d_a, a.data(), matrix_size, 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_mem_pool.d_b, b.data(), matrix_size, 
                    cudaMemcpyHostToDevice, stream);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(g_cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_T,
                n,
                n,
                n,
                &alpha,
                g_mem_pool.d_a, n,
                g_mem_pool.d_b, n,
                &beta,
                g_mem_pool.d_c, n);
    
    cudaMemcpyAsync(c.data(), g_mem_pool.d_c, matrix_size,
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return c;
}

void CleanupCUBLAS() {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
    g_mem_pool.free();
}