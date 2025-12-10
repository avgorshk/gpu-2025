#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>


std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float* matrix1;
    float* matrix2;
    float* res_matrix;
    const size_t memory = n * n * sizeof(float);
    cudaMalloc(&matrix1, memory);
    cudaMalloc(&matrix2, memory);
    cudaMalloc(&res_matrix, memory);
    cudaMemcpy(matrix1, a.data(), memory, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix2, b.data(), memory, cudaMemcpyHostToDevice);
    float a_coef = 1.0f;
    float b_coef = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &a_coef, matrix2, n, matrix1, n, &b_coef, res_matrix, n);
    std::vector<float> matmul_res(n * n);
    cudaMemcpy(matmul_res.data(), res_matrix, memory, cudaMemcpyDeviceToHost);
    cudaFree(matrix1);
    cudaFree(matrix2);
    cudaFree(res_matrix);
    cublasDestroy(handle);
    return matmul_res;
}