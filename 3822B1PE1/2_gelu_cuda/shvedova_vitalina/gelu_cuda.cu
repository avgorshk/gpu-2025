#include "gelu_cuda.h"
#include <cuda.h>
#include <cmath>
#include <iostream>

__global__ void gelu_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = in[idx];
    const float kAlpha = 0.7978845608f;
    float x3 = x * x * x;
    float inner = kAlpha * (x + 0.044715f * x3);
    float exp_term = __expf(-2.0f * inner);
    float tanh_val = (1.0f - exp_term) / (1.0f + exp_term);
    out[idx] = 0.5f * x * (1.0f + tanh_val);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) return {};
    const int n = static_cast<int>(input.size());
    std::vector<float> output(n);
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc((void**)&d_in, n * sizeof(float));
    cudaMalloc((void**)&d_out, n * sizeof(float));
    
    float* h_in_pinned = nullptr;
    float* h_out_pinned = nullptr;
    cudaMallocHost((void**)&h_in_pinned, n * sizeof(float));
    cudaMallocHost((void**)&h_out_pinned, n * sizeof(float));
    memcpy(h_in_pinned, input.data(), n * sizeof(float));
    cudaMemcpyAsync(d_in, h_in_pinned, n * sizeof(float), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    gelu_kernel<<<gridSize, blockSize>>>(d_in, d_out, n);

    cudaMemcpyAsync(h_out_pinned, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    memcpy(output.data(), h_out_pinned, n * sizeof(float));

    cudaFreeHost(h_in_pinned);
    cudaFreeHost(h_out_pinned);
    cudaFree(d_in);
    cudaFree(d_out);
    
    return output;
}
