#include <cuda_runtime.h>
#include <cufft.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#include "fft_cufft.h"

__global__ void normalize_kernel(cufftComplex* data, int total_elems,
                                 float inv_n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elems) {
    data[idx].x *= inv_n;
    data[idx].y *= inv_n;
  }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
  const int total_floats = static_cast<int>(input.size());

  if (batch <= 0 || total_floats == 0 || (total_floats % (2 * batch)) != 0) {
    return {};
  }

  const int n = total_floats / (2 * batch);
  const int total_complex = n * batch;
  const size_t bytes =
      static_cast<size_t>(total_complex) * sizeof(cufftComplex);

  static cufftComplex* d_data = nullptr;
  static int capacity_complex = 0;

  static cufftHandle plan = 0;
  static int plan_n = 0;
  static int plan_batch = 0;

  static cudaStream_t stream = nullptr;

  if (!stream) {
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      throw std::runtime_error("cudaStreamCreate failed in FffCUFFT");
    }
  }

  if (total_complex > capacity_complex) {
    if (d_data) {
      cudaFree(d_data);
      d_data = nullptr;
    }

    if (cudaMalloc(&d_data, bytes) != cudaSuccess) {
      capacity_complex = 0;
      throw std::runtime_error("cudaMalloc failed in FffCUFFT");
    }
    capacity_complex = total_complex;
  }

  if (!plan || plan_n != n || plan_batch != batch) {
    if (plan) {
      cufftDestroy(plan);
      plan = 0;
    }

    int rank = 1;
    int n_arr[1] = {n};
    int istride = 1, ostride = 1;
    int idist = n, odist = n;
    int inembed[1] = {n};
    int onembed[1] = {n};

    cufftResult res = cufftPlanMany(&plan, rank, n_arr, inembed, istride, idist,
                                    onembed, ostride, odist, CUFFT_C2C, batch);

    if (res != CUFFT_SUCCESS) {
      throw std::runtime_error("cufftPlanMany failed in FffCUFFT");
    }

    cufftSetStream(plan, stream);

    plan_n = n;
    plan_batch = batch;
  }

  std::vector<cufftComplex> h_data(total_complex);
  for (int i = 0; i < total_complex; ++i) {
    int in_idx = 2 * i;
    h_data[i].x = input[in_idx];
    h_data[i].y = input[in_idx + 1];
  }

  cudaMemcpyAsync(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice, stream);

  {
    cufftResult res = cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    if (res != CUFFT_SUCCESS) {
      throw std::runtime_error("cufftExecC2C FORWARD failed in FffCUFFT");
    }
  }

  {
    cufftResult res = cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    if (res != CUFFT_SUCCESS) {
      throw std::runtime_error("cufftExecC2C INVERSE failed in FffCUFFT");
    }
  }

  float inv_n = 1.0f / static_cast<float>(n);
  int block = 256;
  int grid = (total_complex + block - 1) / block;
  normalize_kernel<<<grid, block, 0, stream>>>(d_data, total_complex, inv_n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("normalize_kernel launch failed in FffCUFFT");
  }

  cudaMemcpyAsync(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  std::vector<float> output(static_cast<size_t>(total_floats));
  for (int i = 0; i < total_complex; ++i) {
    int out_idx = 2 * i;
    output[out_idx] = h_data[i].x;
    output[out_idx + 1] = h_data[i].y;
  }

  return output;
}
