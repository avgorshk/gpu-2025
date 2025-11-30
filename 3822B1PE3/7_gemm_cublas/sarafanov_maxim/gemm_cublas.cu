#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS_ERROR(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("Input matrices must have size n*n");
    }

    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS_ERROR(cublasSgemm(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n, n, n,
                                  &alpha,
                                  d_B, n,
                                  d_A, n,
                                  &beta,
                                  d_C, n));

    std::vector<float> c(n * n);
    CHECK_CUDA_ERROR(cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));

    return c;
}