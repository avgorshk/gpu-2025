#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
                          
    int pow_n = n * n;
    int d_sz = pow_n * sizeof(float);  
    float *d_a, *d_b, *d_c;
    std::vector<float> c(pow_n, 0.0);


    cudaMalloc(&d_a, d_sz);
    cudaMalloc(&d_b, d_sz);
    cudaMalloc(&d_c, d_sz);
    cudaMemcpy(d_a, a.data(), d_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), d_sz, cudaMemcpyHostToDevice);
 

    cublasHandle_t handle;
    cublasCreate(&handle);

    float al = 1.0, bt = 0.0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, &al, d_b , n,
                d_a, n, &bt, d_c, n);
 

    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, d_sz, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);


    return c;
}