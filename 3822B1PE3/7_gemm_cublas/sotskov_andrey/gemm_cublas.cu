#include "gemm_cublas.h"
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float> &matrixA,
                              const std::vector<float> &matrixB, int matrixSize)
{
    std::vector<float> resultMatrix(matrixSize * matrixSize);

    float *deviceMatrixA = nullptr;
    float *deviceMatrixB = nullptr;
    float *deviceMatrixC = nullptr;

    const size_t matrixBytes = matrixSize * matrixSize * sizeof(float);

    cudaMalloc(&deviceMatrixA, matrixBytes);
    cudaMalloc(&deviceMatrixB, matrixBytes);
    cudaMalloc(&deviceMatrixC, matrixBytes);

    cudaMemcpy(deviceMatrixA, matrixA.data(), matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, matrixB.data(), matrixBytes, cudaMemcpyHostToDevice);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                matrixSize, matrixSize, matrixSize,
                &alpha,
                deviceMatrixB, matrixSize,
                deviceMatrixA, matrixSize,
                &beta,
                deviceMatrixC, matrixSize);

    cudaMemcpy(resultMatrix.data(), deviceMatrixC, matrixBytes, cudaMemcpyDeviceToHost);

    cublasDestroy(cublasHandle);
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);

    return resultMatrix;
}