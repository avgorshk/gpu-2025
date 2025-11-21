#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("Input matrices must have size n*n");
    }

    cublasStatus_t status;
    cudaError_t cudaStatus;
    
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS handle creation failed");
    }

    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    
    cudaStatus = cudaMalloc(&d_A, size);
    if (cudaStatus != cudaSuccess) {
        cublasDestroy(handle);
        throw std::runtime_error("CUDA memory allocation for A failed");
    }
    
    cudaStatus = cudaMalloc(&d_B, size);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_A);
        cublasDestroy(handle);
        throw std::runtime_error("CUDA memory allocation for B failed");
    }
    
    cudaStatus = cudaMalloc(&d_C, size);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cublasDestroy(handle);
        throw std::runtime_error("CUDA memory allocation for C failed");
    }

    try {
        status = cublasSetMatrix(n, n, sizeof(float), a.data(), n, d_A, n);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to set matrix A on device");
        }
        
        status = cublasSetMatrix(n, n, sizeof(float), b.data(), n, d_B, n);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to set matrix B on device");
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        status = cublasSgemm(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n, n, n,
                            &alpha,
                            d_B, n,
                            d_A, n,
                            &beta,
                            d_C, n);
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS SGEMM operation failed");
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("CUDA device synchronization failed");
        }

        std::vector<float> c(n * n);
        status = cublasGetMatrix(n, n, sizeof(float), d_C, n, c.data(), n);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to get result matrix from device");
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);

        return c;

    } catch (const std::exception& e) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        throw;
    }
}