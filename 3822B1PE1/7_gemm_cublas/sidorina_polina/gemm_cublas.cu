#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

struct GemmParams
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int matrix_size = 0;
};

void ComputeGemmCublas(const float* matrix_a,
                       const float* matrix_b,
                       float* matrix_c,
                       const GemmParams& params)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemm(handle,
                CUBLAS_OP_T,
                CUBLAS_OP_T,
                params.matrix_size,
                params.matrix_size,
                params.matrix_size,
                &params.alpha,
                matrix_b,
                params.matrix_size,
                matrix_a,
                params.matrix_size,
                &params.beta,
                matrix_c,
                params.matrix_size);
    
    cublasDestroy(handle);
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    GemmParams params;
    params.matrix_size = n;
    
    const int total_elements = n * n;
    const size_t data_size_bytes = total_elements * sizeof(float);
    
    std::vector<float> result(total_elements, 0.0f);
    
    float* d_matrix_a = nullptr;
    float* d_matrix_b = nullptr;
    float* d_matrix_c = nullptr;
    
    cudaMalloc(&d_matrix_a, data_size_bytes);
    cudaMalloc(&d_matrix_b, data_size_bytes);
    cudaMalloc(&d_matrix_c, data_size_bytes);
    
    cudaMemcpy(d_matrix_a, a.data(), data_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, b.data(), data_size_bytes, cudaMemcpyHostToDevice);
    
    cudaMemset(d_matrix_c, 0, data_size_bytes);
    
    ComputeGemmCublas(d_matrix_a, d_matrix_b, d_matrix_c, params);
    
    cudaMemcpy(result.data(), d_matrix_c, data_size_bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);
    
    return result;
}