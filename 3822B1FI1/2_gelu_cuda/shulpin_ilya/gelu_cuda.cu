#include "gelu_cuda.h"

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

__device__ static inline float gelu_scalar_exp(float x) {
    constexpr float c = 0.044715f;
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    float x3 = x * x * x;
    float z = sqrt_2_over_pi * (x + c * x3);
    // GELU ≈ x * sigmoid(2*z) = x / (1 + exp(-2*z))
    float s = 1.0f / (1.0f + std::exp(-2.0f * z));
    return x * s;
}

__global__ void kernel(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

    for (size_t i = idx; i < n; i += stride) {
        out[i] = gelu_scalar_exp(in[i]);
    }
}


std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();

    if (n == 0) {
        return {};
    }

    size_t bytes = n * sizeof(float);
    float *d_in = nullptr;
    float *d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    size_t blocks_needed = (n + threads - 1) / threads;
    size_t blocks = std::min<size_t>(blocks_needed, 65535);

    kernel<<<blocks, threads>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output(n);
    CUDA_CHECK(cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return output;
}
