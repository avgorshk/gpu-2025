#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

constexpr int cuda_block_size = 64;
using std::vector;

__global__ void MatMul(const float* a, const float* b, float* ans, int n) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float res = 0.0f;
        for (int k = 0; k < n; ++k) {
            res += a[row * n + k] * b[k * n + col];
        }
        ans[row * n + col] = res;
    }
}

__host__ vector<float> NaiveGemmCUDA(const vector<float>& a,
                                     const vector<float>& b,
                                     int n) {
    int req_mem = n * n * sizeof(float);
    float* in1, *in2, *ans;
	vector<float> result(n * n);

    cudaMalloc(&in1, req_mem);
    cudaMalloc(&in2, req_mem);
    cudaMalloc(&ans, req_mem);

    cudaMemcpy(in1, a.data(), req_mem, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(in2, b.data(), req_mem, cudaMemcpyKind::cudaMemcpyHostToDevice);
    
    int dim_size = (n + cuda_block_size - 1) / cuda_block_size;
    dim3 kernel_size(dim_size, dim_size);
    dim3 block(cuda_block_size, cuda_block_size);

    MatMul<<<kernel_size, block>>> (in1, in2, ans, n);
    cudaMemcpy(result.data(), ans, req_mem, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(in1);
    cudaFree(in2);
    cudaFree(ans);

    return result;
}
