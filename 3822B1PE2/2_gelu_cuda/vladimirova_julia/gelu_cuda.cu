#include "gelu_cuda.h"

__constant__ float sqrt2p = 0.7978845608028654;
__constant__ float cf = 0.044715;

__global__ void work_frm_gelu(float* __restrict__ output, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t<n) {
        float x = output[t];
        output[t] *= 0.5 * (1.0 + tanhf(sqrt2p * (x + cf * x*x*x)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  
  size_t n = input.size();
  std::vector<float> output(n);

  float* d_output;
  cudaMalloc(&d_output, n * sizeof(float));
  cudaMemcpy(d_output, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;

  work_frm_gelu<<<numBlocks, blockSize>>>(d_output, n);
  cudaDeviceSynchronize();

  cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_output);

  return output;
    // Place your implementation here
}