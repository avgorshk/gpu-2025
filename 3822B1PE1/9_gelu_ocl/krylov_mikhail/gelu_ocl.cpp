#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <iostream>
#include <mutex>
#include <vector>

namespace {
static const char* kGELUKernelSource = R"(
__kernel void gelu(__global const float* __restrict__ input, __global float* __restrict__ output, int n) {
  const int idx = get_global_id(0);
  if (idx < n) {
    const float x = input[idx];
    output[idx] = x / (1.f + exp(-2.f * 0.797884560802f * (x + 0.044715f * (x * x * x))));
  }
}
)";

template <typename Vec>
void ResizeUninitialized(Vec& v, std::size_t size) {
  struct stub {
    typename Vec::value_type v;
    stub() {}
  };
  reinterpret_cast<std::vector<stub>&>(v).resize(size);
}

cl::Context ctx;
cl::CommandQueue queue;
cl::Kernel kernel;

// just in case GeluOCL is being benchmarked without restarting
void InitializeCL(int platform_ord) {
  static int initialized_platform_ord{-1};
  if (initialized_platform_ord == platform_ord) {
    return;
  }

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  cl::Platform platform = platforms[platform_ord];

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

  ctx = cl::Context(devices);
  queue = cl::CommandQueue(ctx, devices[0]);

  cl::Program program(ctx, kGELUKernelSource);
  program.build(devices);
  kernel = cl::Kernel(program, "gelu");

  initialized_platform_ord = platform_ord;
}
}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
  const std::size_t n = input.size();

  InitializeCL(platform);

  cl::Buffer input_buf(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float),
                       const_cast<float*>(input.data()));
  cl::Buffer output_buf(ctx, CL_MEM_WRITE_ONLY, n * sizeof(float));

  kernel.setArg(0, input_buf);
  kernel.setArg(1, output_buf);
  kernel.setArg(2, static_cast<int>(n));

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);
  //
  std::vector<float> output;
  ResizeUninitialized(output, n);
  //
  queue.finish();

  queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, n * sizeof(float), output.data());

  return output;
}