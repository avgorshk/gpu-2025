#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

__global__ void gelu_kernel(const float* src, float* dst, int num_elems) {
    const float sqrt_2_div_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elems) {
        float val = src[tid];
        float cube = val * val * val;
        float inner_val = sqrt_2_div_pi * (val + coeff * cube);
        dst[tid] = 0.5f * val * (1.0f + tanhf(inner_val));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    std::vector<float> out(n);
    if (n == 0) return out;
    float *d_inp, *d_out;
    cudaMalloc(&d_inp, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_inp, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blk_size = 256;
    int blk_count = (n + blk_size - 1) / blk_size;
    gelu_kernel<<<blk_count, blk_size>>>(d_inp, d_out, n);

    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_inp);
    cudaFree(d_out);

    return out;
}