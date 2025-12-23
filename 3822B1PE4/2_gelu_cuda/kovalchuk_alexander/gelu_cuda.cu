#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

__device__ __forceinline__ float gelu_tanh_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    float x2 = x * x;
    float x3 = x2 * x;
    float t = sqrt_2_over_pi * (x + coeff * x3);

    float e2t = expf(2.0f * t);
    float tanh_t = (e2t - 1.0f) / (e2t + 1.0f);

    return 0.5f * x * (1.0f + tanh_t);
}

__global__ void gelu_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = gelu_tanh_approx(in[idx]);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    if (n == 0) {
        return {};
    }

    std::vector<float> output(n);

    float* d_in  = nullptr;
    float* d_out = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_in,  n * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc d_in failed");
    }
    err = cudaMalloc(&d_out, n * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        throw std::runtime_error("cudaMalloc d_out failed");
    }

    err = cudaMemcpy(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        throw std::runtime_error("cudaMemcpy H2D failed");
    }

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    gelu_kernel<<<blocks, threads>>>(d_in, d_out, n);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        throw std::runtime_error("cudaDeviceSynchronize failed");
    }

    err = cudaMemcpy(output.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        throw std::runtime_error("cudaMemcpy D2H failed");
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return output;
}
