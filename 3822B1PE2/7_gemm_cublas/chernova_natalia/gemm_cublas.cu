#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b,
                              int n)
{
    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n))
    {
        std::cerr << "Error: Matrix sizes don't match!" << std::endl;
        return std::vector<float>();
    }

    int num_elements = n * n;
    int byte_size = num_elements * sizeof(float);
    float *d_a, *d_b, *d_c;
    std::vector<float> c(num_elements, 0.0);

    cudaMalloc(&d_a, byte_size);
    cudaMalloc(&d_b, byte_size);
    cudaMalloc(&d_c, byte_size);
    cudaMemcpy(d_a, a.data(), byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), byte_size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0, beta = 0.0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, &alpha, d_b, n,
                d_a, n, &beta, d_c, n);

    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, byte_size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);

    return c;
}