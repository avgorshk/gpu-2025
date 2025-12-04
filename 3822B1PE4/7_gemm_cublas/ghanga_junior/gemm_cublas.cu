#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    const size_t num_elements = static_cast<size_t>(n) * n;
    const size_t size_bytes   = num_elements * sizeof(float);

    // Allocation device
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaMalloc(&d_a, size_bytes);
    cudaMalloc(&d_b, size_bytes);
    cudaMalloc(&d_c, size_bytes);

    // Handle cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Copie host -> device
    cudaMemcpy(d_a, a.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size_bytes, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // C = A * B (les deux en row-major mais testés de la même façon côté prof)
    cublasSgemm(
        handle,
        CUBLAS_OP_N, // op(B)
        CUBLAS_OP_N, // op(A)
        n,           // m
        n,           // n
        n,           // k
        &alpha,
        d_b, n,      // B
        d_a, n,      // A
        &beta,
        d_c, n       // C
    );

    // Récupérer le résultat
    std::vector<float> c(num_elements);
    cudaMemcpy(c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
