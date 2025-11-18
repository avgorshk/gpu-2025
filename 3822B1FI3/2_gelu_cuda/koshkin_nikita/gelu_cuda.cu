#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <algorithm>

__device__ __forceinline__ float fast_tanh_like_cuda(float z) {
    float e = __expf(2.0f * z);
    return (e - 1.0f) / (e + 1.0f);
}

__global__ void gelu_kernel(const float* __restrict__ in,
    float* __restrict__ out,
    int n)
{
    constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float kCubic = 0.044715f;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < n;
        idx += blockDim.x * gridDim.x)
    {
        float x = in[idx];
        float x2 = x * x;
        float x3 = x2 * x;
        float t = kSqrt2OverPi * (x + kCubic * x3);
        float th = fast_tanh_like_cuda(t);
        out[idx] = 0.5f * x * (1.0f + th);
    }
}

static inline void throwOnCudaError(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    std::vector<float> output(n);
    if (n == 0) { 
        return output; 
    }

    float* d_in = nullptr, * d_out = nullptr;

    throwOnCudaError(cudaMalloc((void**)&d_in, n * sizeof(float)), "cudaMalloc d_in");
    throwOnCudaError(cudaMalloc((void**)&d_out, n * sizeof(float)), "cudaMalloc d_out");
    throwOnCudaError(cudaMemcpy(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");

    int block = 256;
    int grid = std::min((n + block - 1) / block, 65535);
    gelu_kernel << <grid, block >> > (d_in, d_out, n);
    throwOnCudaError(cudaGetLastError(), "kernel launch");

    throwOnCudaError(cudaMemcpy(output.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H");

    cudaFree(d_in);
    cudaFree(d_out);

    return output;
}