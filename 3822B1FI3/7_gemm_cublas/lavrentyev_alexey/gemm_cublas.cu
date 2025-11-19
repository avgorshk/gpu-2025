#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

using std::vector;

vector<float> GemmCUBLAS(const vector<float>& a,
                         const vector<float>& b,
                         int n) {
    int size = n * n;
    int req_mem = size * sizeof(float);
    float* in1, *in2, *ans, *transpose;
	vector<float> result(size);
    
    cudaMalloc(&in1, req_mem);
    cudaMalloc(&in2, req_mem);
    cudaMalloc(&ans, req_mem);
    cudaMalloc(&transpose, req_mem);

    cudaMemcpy(in1, a.data(), req_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(in2, b.data(), req_mem, cudaMemcpyHostToDevice);

    cublasHandle_t handler;
    cublasCreate_v2(&handler);
    float one = 1.0f;
    float zero = 0.0f;

    cublasSgemm(handler, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &one, in1, n, in2, n, &zero, ans, n);
    cublasSgeam(handler, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &one, ans, n, &zero, nullptr, n, transpose, n);
    cudaMemcpy(result.data(), transpose, req_mem, cudaMemcpyDeviceToHost);
    
    cudaFree(in1);
    cudaFree(in2);
    cudaFree(ans);
    cudaFree(transpose);

    cublasDestroy(handler);

    return result;
}

int main() {
    
}