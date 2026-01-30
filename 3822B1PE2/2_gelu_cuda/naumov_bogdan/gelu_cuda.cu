#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <vector>

__constant__ float c_sqrt_2_over_pi = 0.7978845608028654f;
__constant__ float c_gelu_alpha = 0.044715f;

__device__ __forceinline__ float a_tanh(float x) {
    if (x > 8.0f) return 1.0f;
    if (x < -8.0f) return -1.0f;
    const float exp_2x = __expf(2.0f * x);
    return (exp_2x - 1.0f) / (exp_2x + 1.0f);
}

__global__ void gelu_kernel(const float* __restrict__ input, 
                           float* __restrict__ output, 
                           const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        float tanh_arg = c_sqrt_2_over_pi * __fmaf_rn(c_gelu_alpha, x * x * x, x);
        output[idx] = __fmaf_rn(0.5f * x, __fadd_rn(1.0f, a_tanh(tanh_arg)), 0.0f);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return std::vector<float>();
    }

    const int input_size = static_cast<int>(input.size());
    std::vector<float> output(input_size);
    float *d_input, *d_output;
    
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));
    
    cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    
    const int block_size = 1024;
    const int num_blocks = (input_size + block_size - 1) / block_size;
    
    gelu_kernel<<<num_blocks, block_size>>>(d_input, d_output, input_size);

    cudaDeviceSynchronize();
    
    cudaMemcpy(output.data(), d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}