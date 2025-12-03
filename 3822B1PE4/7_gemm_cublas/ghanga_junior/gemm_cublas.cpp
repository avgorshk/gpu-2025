#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

std::vector<float> GemmCuBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    const size_t size = static_cast<size_t>(n) * n;
    std::vector<float> c(size, 0.0f);

    if (n <= 0 || a.size() != size || b.size() != size)
        return c;

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = size * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // C = A * B en row-major devient :
    // C^T = B^T * A^T en column-major
    cublasSgemm(
        handle,
        CUBLAS_OP_N, // transpose B ? non
        CUBLAS_OP_N, // transpose A ? non
        n, n, n,
        &alpha,
        d_b, n,
        d_a, n,
        &beta,
        d_c, n
    );

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
