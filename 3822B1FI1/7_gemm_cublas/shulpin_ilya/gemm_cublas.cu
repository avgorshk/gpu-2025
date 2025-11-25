#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t N = static_cast<size_t>(n);

    std::vector C(N * N, 0.0f);

    const size_t bytes = N * N * sizeof(float);

    float* D_a = nullptr, * D_b = nullptr, * D_c = nullptr;

    const float alpha = 1.0f, beta = 0.0f;

    cudaMalloc(&D_a, bytes);
    cudaMalloc(&D_b, bytes);
    cudaMalloc(&D_c, bytes);

    cudaMemcpy(D_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(D_b, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, n, n,
        &alpha,
        D_b, n,
        D_a, n,
        &beta,
        D_c, n);

    cudaMemcpy(C.data(), D_c, bytes, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    cudaFree(D_a);
    cudaFree(D_b);
    cudaFree(D_c);

    return C;
}