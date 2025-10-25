#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) {
        throw std::invalid_argument("Matrix size n must be positive");
    }
    const size_t matrix_size = n * n;
    if (a.size() != matrix_size || b.size() != matrix_size) {
        throw std::invalid_argument("Matrix sizes do not match n*n");
    }

    const size_t bytes = matrix_size * sizeof(float);
    

    std::vector<float> c(matrix_size);

    float* d_a = nullptr;
    float* d_b = nullptr; 
    float* d_c = nullptr;
    
    cudaError_t cuda_status;
    cuda_status = cudaMalloc(&d_a, bytes);
    if (cuda_status != cudaSuccess) throw std::runtime_error("cudaMalloc failed for d_a");
    
    cuda_status = cudaMalloc(&d_b, bytes);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_a);
        throw std::runtime_error("cudaMalloc failed for d_b");
    }
    
    cuda_status = cudaMalloc(&d_c, bytes);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        throw std::runtime_error("cudaMalloc failed for d_c");
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    cudaStreamSynchronize(stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, 
                CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_a, n,
                d_b, n,  
                &beta,
                d_c, n);

    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}