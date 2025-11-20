#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    int memory = n * n * sizeof(float);
    int square = n * n;
    float* in1, *in2, *out, *out_transposed;
	std::vector<float> result(square);
    
    cudaMalloc(&in1, memory);
    cudaMalloc(&in2, memory);
    cudaMalloc(&out, memory);
    cudaMalloc(&out_transposed, memory);

    cudaMemcpy(in1, a.data(), memory, cudaMemcpyHostToDevice);
    cudaMemcpy(in2, b.data(), memory, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    float alpha = 1.0f, beta= 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, in1, n, in2, n, &beta, out, n);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, out, n, &beta, nullptr, n, out_transposed, n);
    cudaMemcpy(result.data(), out_transposed, memory, cudaMemcpyDeviceToHost);
    
    cudaFree(in1);
    cudaFree(in2);
    cudaFree(out);
    cudaFree(out_transposed);
    cublasDestroy(handle);
    return result;
}