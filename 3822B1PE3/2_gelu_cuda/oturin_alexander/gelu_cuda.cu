#include "gelu_cuda.h"

__global__ void GeluKernel(const float* input, float* output, int size, float sqrt_2_pi_m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        float x = input[i];
        float x_cubed = x * x * x;
        float expo = expf(sqrt_2_pi_m * (x + 0.044715f * x_cubed));
        output[i] = x * expo / (expo + 1.0f);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = input.size();
    std::vector<float> output(size);
    
    if (size == 0)
        return output;
    
    const float sqrt_2_pi_m = std::sqrt(2.0f / CUDART_PI_F) * 2;
    
    float *dev_input, *dev_output;
    cudaMalloc(&dev_input, size * sizeof(float));
    cudaMalloc(&dev_output, size * sizeof(float));
    
    cudaMemcpy(dev_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int block_count = (size + block_size - 1) / block_size;
    
    GeluKernel<<<block_count, block_size>>>(dev_input, dev_output, size, sqrt_2_pi_m);
    
    cudaMemcpy(output.data(), dev_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_input);
    cudaFree(dev_output);
    
    return output;
}