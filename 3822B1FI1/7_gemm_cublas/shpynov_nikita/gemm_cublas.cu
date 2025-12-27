#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#define CUDA_CHECK(call)                                         \
    do                                                           \
    {                                                            \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess)                                  \
        {                                                        \
            printf("CUDA error %s:%d: %s\n",                     \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                             \
        }                                                        \
    } while (0)

#define CUBLAS_CHECK(call)                     \
    do                                         \
    {                                          \
        cublasStatus_t stat = call;            \
        if (stat != CUBLAS_STATUS_SUCCESS)     \
        {                                      \
            printf("CUBLAS error %s:%d: %d\n", \
                   __FILE__, __LINE__, stat);  \
            exit(1);                           \
        }                                      \
    } while (0)

std::vector<float> rowToColumn(const std::vector<float> &src, int n)
{
    std::vector<float> dst(n * n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            dst[j * n + i] = src[i * n + j];
        }
    }
    return dst;
}
std::vector<float> columnToRow(const std::vector<float> &src, int n)
{
    std::vector<float> dst(n * n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            dst[i * n + j] = src[j * n + i];
        }
    }

    return dst;
}
std::vector<float> GemmCUBLAS(const std::vector<float> &A,
                              const std::vector<float> &B,
                              int n)
{
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::vector<float> A_col = rowToColumn(A, n);
    std::vector<float> B_col = rowToColumn(B, n);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t matrixSize = n * n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, matrixSize));
    CUDA_CHECK(cudaMalloc(&d_B, matrixSize));
    CUDA_CHECK(cudaMalloc(&d_C, matrixSize));

    CUDA_CHECK(cudaMemcpy(d_A, A_col.data(), matrixSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B_col.data(), matrixSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, matrixSize));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n,
                    &alpha,
                    d_A, n,
                    d_B, n,
                    &beta,
                    d_C, n));

    std::vector<float> C_col(n * n);
    CUDA_CHECK(cudaMemcpy(C_col.data(), d_C, matrixSize, cudaMemcpyDeviceToHost));
    std::vector<float> C = columnToRow(C_col, n);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(handle);

    return C;
}