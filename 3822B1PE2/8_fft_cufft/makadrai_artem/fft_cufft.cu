#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>

__global__ void normalize_kernel(cufftComplex *data, float scale, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx].x *= scale;
    data[idx].y *= scale;
  }
}

static cufftHandle plan = 0;
static cufftComplex *d_data = nullptr;
static cudaStream_t stream = nullptr;
static int cached_n = 0;
static int cached_batch = 0;
static size_t allocated_size = 0;
static bool initialized = false;

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch) {
  if (input.empty() || batch <= 0) {
    return std::vector<float>();
  }

  int n = input.size() / (2 * batch);
  if (n <= 0) {
    return std::vector<float>();
  }

  int total_complex = n * batch;
  size_t bytes = total_complex * sizeof(cufftComplex);

  if (!initialized) {
    cudaStreamCreate(&stream);
    initialized = true;
  }

  if (plan == 0 || cached_n != n || cached_batch != batch) {
    if (plan != 0) {
      cufftDestroy(plan);
    }
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    cufftSetStream(plan, stream);
    cached_n = n;
    cached_batch = batch;
  }

  if (d_data == nullptr || allocated_size < bytes) {
    if (d_data != nullptr) {
      cudaFree(d_data);
    }
    cudaMalloc(&d_data, bytes);
    allocated_size = bytes;
  }

  std::vector<float> output(input.size());

  cudaMemcpyAsync(d_data, input.data(), bytes, cudaMemcpyHostToDevice, stream);

  cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
  cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

  float scale = 1.0f / static_cast<float>(n);
  int blockSize = 256;
  int numBlocks = (total_complex + blockSize - 1) / blockSize;
  normalize_kernel<<<numBlocks, blockSize, 0, stream>>>(d_data, scale,
                                                        total_complex);

  cudaMemcpyAsync(output.data(), d_data, bytes, cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  return output;
}
