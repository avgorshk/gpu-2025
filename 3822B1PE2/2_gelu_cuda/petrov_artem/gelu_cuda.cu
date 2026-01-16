#include "gelu_cuda.h"
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* input, float* output, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = input[i];
    float x3 = x * x * x;
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float GELU_COEFF = 0.044715f;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    
    float exp_val = __expf(2.0f * inner);
    float tanh_val = (exp_val - 1.0f) / (exp_val + 1.0f);
    
    output[i] = 0.5f * x * (1.0f + tanh_val);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);
    
    if (n == 0) return output;

    float *d_input, *d_output;
    size_t bytes = n * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    gelu_kernel<<<grid_size, block_size>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
