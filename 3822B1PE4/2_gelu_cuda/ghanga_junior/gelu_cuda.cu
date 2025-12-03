#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <vector>

// Kernel CUDA : applique GELU à chaque élément
__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float c      = 0.7978845608f;   // sqrt(2/pi)
    const float alpha  = 0.044715f;

    float x  = input[idx];
    float x2 = x * x;
    float x3 = x2 * x;

    float v  = x + alpha * x3;
    float z  = c * v;

    // tanh(z) ≈ (1 - e^{-2z}) / (1 + e^{-2z})
    float e  = __expf(-2.0f * z);         // version rapide de exp pour float
    float t  = (1.0f - e) / (1.0f + e);

    output[idx] = 0.5f * x * (1.0f + t);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    std::vector<float> output(n);

    if (n == 0) {
        return output;
    }

    // Pointeurs device
    float* d_input  = nullptr;
    float* d_output = nullptr;

    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // Allocation GPU
    cudaMalloc(&d_input,  bytes);
    cudaMalloc(&d_output, bytes);

    // Copie host -> device
    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    // Configuration du kernel
    const int blockSize = 256;
    const int gridSize  = (n + blockSize - 1) / blockSize;

    // Lancement du kernel
    gelu_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    // Attendre la fin du GPU
    cudaDeviceSynchronize();

    // Copie device -> host
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    // Libération mémoire GPU
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
