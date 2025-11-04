#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gelu_cuda.h"

namespace {
constexpr float kSQRT2DPI = 0.797884560802f;
__global__ void GELU_kernel(const float4* __restrict__ in4, float4* __restrict__ out4, std::size_t n) {
  const auto compute = [](float x) -> float {
    return 0.5f * x * (1.0f + tanhf(kSQRT2DPI * fmaf(0.044715f, x * x * x, x)));
  };

  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (__builtin_expect(idx < n, 1)) {
    out4[idx].x = compute(in4[idx].x);
    out4[idx].y = compute(in4[idx].y);
    out4[idx].z = compute(in4[idx].z);
    out4[idx].w = compute(in4[idx].w);
  }
}

template <typename Vec>
void ResizeUninitialized(Vec& v, std::size_t size) {
  struct stub {
    typename Vec::value_type v;
    stub() {}
  };
  reinterpret_cast<std::vector<stub>&>(v).resize(size);
}
}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  const std::size_t n = input.size();

  std::vector<float> output;
  ResizeUninitialized(output, n);

  cudaHostRegister((void*)input.data(), n * sizeof(float), cudaHostRegisterDefault);
  cudaHostRegister((void*)output.data(), n * sizeof(float), cudaHostRegisterDefault);

  int minGridSize, blockSize;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, GELU_kernel, 0, 0);

  // n = 134'217'728
  const std::size_t n4 = n / (sizeof(float4) / sizeof(float));  // n/(4) = 33'554'432
  constexpr std::size_t chunks4 = 4;
  const std::size_t kChunkSize4 = n4 / chunks4;
  const int gridSize = (n4 + blockSize - 1) / blockSize;
  //
  float4* devbuf;
  cudaMalloc(&devbuf, 2 * 2 * kChunkSize4 * sizeof(float4));  // 2 x (in, out)
  //
  struct {
    cudaStream_t stream;
    float4 *in, *out;
  } chunk_handles[2];
  {
    float4* p_devbuf{devbuf};
    const auto next_devbuf_chunk_partition = [&]() {
      auto* p = p_devbuf;
      p_devbuf += kChunkSize4;
      return p;
    };
    for (std::size_t i = 0; i < std::size(chunk_handles); ++i) {
      auto& handle = chunk_handles[i];
      cudaStreamCreate(&handle.stream);
      handle.in = next_devbuf_chunk_partition();
      handle.out = next_devbuf_chunk_partition();
    }
  }
  {
    auto* active_handle{chunk_handles + 0};
    auto* awaiting_handle{chunk_handles + 1};

    // no need in managing leftovers as n = 134'217'728 (?)
    std::size_t off{0};
    const auto kChunkSize{kChunkSize4 * (sizeof(float4) / sizeof(float))};  // as tied to the effective out buf
    for (std::size_t i = 0; i < chunks4; ++i, off += kChunkSize) {
      auto& handle = *active_handle;

      cudaMemcpyAsync(handle.in, input.data() + off, kChunkSize4 * sizeof(float4), cudaMemcpyHostToDevice,
                      handle.stream);
      GELU_kernel<<<gridSize, blockSize, 0, handle.stream>>>(handle.in, handle.out, kChunkSize4);
      cudaMemcpyAsync(output.data() + off, handle.out, kChunkSize4 * sizeof(float4), cudaMemcpyDeviceToHost,
                      handle.stream);

      std::swap(active_handle, awaiting_handle);
    }
  }

  for (std::size_t i = 0; i < std::size(chunk_handles); ++i) {
    cudaStreamSynchronize(chunk_handles[i].stream);
    cudaStreamDestroy(chunk_handles[i].stream);
  }
  cudaFree(devbuf);

  cudaHostUnregister((void*)input.data());
  cudaHostUnregister((void*)output.data());

  return output;
}
