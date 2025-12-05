#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

using std::vector;

vector<float> GemmCUBLAS(const vector<float>& a,
                         const vector<float>& b,
                         int n) {
    int size = n * n;
    int req_mem = size * sizeof(float);
    float* in1, *in2, *ans;
	vector<float> result(size, 0.0f);
    
    cudaMalloc(&in1, req_mem);
    cudaMalloc(&in2, req_mem);
    cudaMalloc(&ans, req_mem);

    cudaMemcpy(in1, a.data(), req_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(in2, b.data(), req_mem, cudaMemcpyHostToDevice);

    cublasHandle_t handler;
    cublasCreate_v2(&handler);
    float one = 1.0f;
    float zero = 0.0f;

    cublasSgemm(handler, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, in2, n, in1, n, &zero, ans, n);

    cudaMemcpy(result.data(), ans, req_mem, cudaMemcpyDeviceToHost);
    
    cudaFree(in1);
    cudaFree(in2);
    cudaFree(ans);

    cublasDestroy(handler);

    return result;
}
