#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

using std::vector;
const int block_size = 16;

__global__ void multiplyMatrix(const float* in1, const float* in2, float* ans, int n) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    
    __shared__ float a[block_size][block_size];
    __shared__ float b[block_size][block_size];

    float res = 0.0f;

    for (int k = 0; k < (n + block_size - 1) / block_size; ++k) {

        a[threadIdx.y][threadIdx.x] = in1[row * n + k * block_size + threadIdx.x];
        b[threadIdx.y][threadIdx.x] = in2[(k * block_size + threadIdx.y) * n + col];
        
        __syncthreads();
        for (int t = 0; t < block_size; ++t) {
            res += a[threadIdx.y][t] * b[t][threadIdx.x];
        }
        __syncthreads();  
    }

    ans[row * n + col] = res;
}

__host__ vector<float> BlockGemmCUDA(const vector<float>& a,
                                     const vector<float>& b,
                                     int n) {
    int size = n * n;
    int req_mem = size * sizeof(float);
    float* in1, *in2, *ans;
	vector<float> result(size);
    
    cudaMalloc(&in1, req_mem);
    cudaMalloc(&in2, req_mem);
    cudaMalloc(&ans, req_mem);

    cudaMemcpy(in1, a.data(), req_mem, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(in2, b.data(), req_mem, cudaMemcpyKind::cudaMemcpyHostToDevice);
    
    int grid = (n + block_size - 1) / block_size;
    dim3 kernel(grid, grid);
    dim3 block(block_size, block_size);
    
    multiplyMatrix<<<kernel, block>>> (in1, in2, ans, n);
    cudaMemcpy(result.data(), ans, req_mem, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(in1);
    cudaFree(in2);
    cudaFree(ans);

    return result;
}
