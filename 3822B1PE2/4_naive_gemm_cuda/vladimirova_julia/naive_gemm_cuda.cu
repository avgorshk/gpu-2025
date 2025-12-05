#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

__global__ void MultMatrEl(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i >= n) || (j >= n)) return;

    float sum = 0.0;
    for (int k = 0; k < n; k++) {
        sum += a[i*n + k] * b[k*n + j];
    }
    c[i*n + j] = sum;
}


std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    int pw_n = n * n;
    int dt_s = pw_n * sizeof(float);
    float *d_a, *d_b, *d_c;
    std::vector<float> c(pw_n);


    cudaMalloc(&d_a, dt_s);
    cudaMalloc(&d_b, dt_s);
    cudaMalloc(&d_c, dt_s);
    cudaMemcpy(d_a, a.data(), dt_s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), dt_s, cudaMemcpyHostToDevice);



    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);
    MultMatrEl<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    

    cudaMemcpy(c.data(), d_c, dt_s, cudaMemcpyDeviceToHost);
    

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);



    return c;
}