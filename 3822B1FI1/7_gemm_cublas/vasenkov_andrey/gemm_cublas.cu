#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& matA,
                              const std::vector<float>& matB,
                              int N) {
    int totalElems = N * N;
    int bytes = totalElems * sizeof(float);
    float *devA, *devB, *devC;
    std::vector<float> result(totalElems, 0.0f);

    cudaMalloc(&devA, bytes);
    cudaMalloc(&devB, bytes);
    cudaMalloc(&devC, bytes);
    cudaMemcpy(devA, matA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, matB.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, devB, N,
                devA, N, &beta, devC, N);

    cudaDeviceSynchronize();
    cudaMemcpy(result.data(), devC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cublasDestroy(cublasHandle);

    return result;
}
