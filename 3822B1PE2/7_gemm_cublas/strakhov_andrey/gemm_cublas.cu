#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b,
                              int n)
{

    size_t size = n * n * sizeof(float);
    float *local_a, *local_b, *local_c;
    std::vector<float> c(n * n, 0.0);

    cudaMalloc(&local_a, size);
    cudaMalloc(&local_b, size);
    cudaMalloc(&local_c, size);
    cudaMemcpy(local_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(local_b, b.data(), size, cudaMemcpyHostToDevice);
    cublasHandle_t handle;
    cublasCreate(&handle);

    float al = 1.0;
    float bt = 0.0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, local_b, n, local_a, n, &bt, local_c, n);

    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), local_c, size, cudaMemcpyDeviceToHost);

    cudaFree(local_a);
    cudaFree(local_b);
    cudaFree(local_c);
    cublasDestroy(handle);

    return c;
}