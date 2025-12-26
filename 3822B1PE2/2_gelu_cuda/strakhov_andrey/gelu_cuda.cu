#include "gelu_cuda.h"

__global__ void kernel(float *__restrict__ out, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float M_PI = 3.14159265358979323846f;
    float calcCoef = sqrt(2.0f / M_PI);
    if (index < size)
    {
        float x = out[index];
        out[index] *= 0.5f * x * (1.0f + tanh(calcCoef * (x + 0.044715f * x * x * x)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float> &input)
{

    int size = input.size();
    std::vector<float> out(size);
    float *d_out;

    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_out, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    kernel<<<numBlocks, blockSize>>>(d_out, size);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    return out;
}
