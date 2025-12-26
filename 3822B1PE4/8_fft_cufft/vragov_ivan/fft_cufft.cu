#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

// Kernel to normalize the result of IFFT
// Divides every element (Real and Imag parts) by n
__global__ void NormalizeKernel(float *data, int total_elements, float factor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    data[idx] *= factor;
  }
}

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch) {
  // Input size is 2 * n * batch (Complex numbers)
  // We can infer n from input.size() and batch
  size_t total_floats = input.size();
  size_t complex_elements_per_batch = (total_floats / 2) / batch;
  int n = static_cast<int>(complex_elements_per_batch);

  // 1. Allocate device memory
  size_t sizeBytes = total_floats * sizeof(float);
  cufftComplex *d_data;
  cudaMalloc((void **)&d_data, sizeBytes);

  // 2. Copy input to device
  cudaMemcpy(d_data, input.data(), sizeBytes, cudaMemcpyHostToDevice);

  // 3. Create cuFFT Plan
  // 1D complex-to-complex transform of size n, batched
  cufftHandle plan;
  cufftPlan1d(&plan, n, CUFFT_C2C, batch);

  // 4. Execute Forward Transform (in-place)
  cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

  // 5. Execute Inverse Transform (in-place)
  cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

  // 6. Normalize
  // cuFFT inverse is unscaled, so result is scaled by n. We must divide by n.
  int threads = 256;
  int blocks = (total_floats + threads - 1) / threads;
  // Treating the array as just a flat array of floats for normalization
  NormalizeKernel<<<blocks, threads>>>((float *)d_data, total_floats, 1.0f / n);
  cudaDeviceSynchronize();

  // 7. Copy result back
  std::vector<float> output(total_floats);
  cudaMemcpy(output.data(), d_data, sizeBytes, cudaMemcpyDeviceToHost);

  // 8. Cleanup
  cufftDestroy(plan);
  cudaFree(d_data);

  return output;
}
