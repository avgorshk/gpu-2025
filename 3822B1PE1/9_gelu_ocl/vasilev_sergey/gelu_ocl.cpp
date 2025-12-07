#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <cstddef>
#include <vector>

namespace
{

  const char *kGeluKernelSrc = R"CLC(
__kernel void gelu_kernel(__global const float* __restrict input,
                          __global float* __restrict output,
                          int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;

  const float k = 1.595769122f; // 2 * sqrt(2/pi)
  const float c = 0.044715f;

  float x  = input[idx];
  float x3 = x * x * x;
  float z  = k * (x + c * x3);
  float s  = 1.0f / (1.0f + exp(-z));
  output[idx] = x * s;
}
)CLC";

  cl::Context g_ctx;
  cl::CommandQueue g_queue;
  cl::Kernel g_kernel;
  int g_inited_platform = -1;

  void InitializeCL(int platform_ord)
  {
    if (g_inited_platform == platform_ord)
    {
      return;
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl::Platform platform = platforms[static_cast<std::size_t>(platform_ord)];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    g_ctx = cl::Context(devices);
    g_queue = cl::CommandQueue(g_ctx, devices[0]);

    cl::Program program(g_ctx, kGeluKernelSrc);
    program.build(devices);

    g_kernel = cl::Kernel(program, "gelu_kernel");

    g_inited_platform = platform_ord;
  }

} // namespace

std::vector<float> GeluOCL(const std::vector<float> &input, int platform)
{
  const std::size_t n = input.size();
  std::vector<float> output(n);
  if (n == 0)
  {
    return output;
  }

  InitializeCL(platform);

  cl::Buffer buf_in(g_ctx,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    n * sizeof(float),
                    const_cast<float *>(input.data()));

  cl::Buffer buf_out(g_ctx,
                     CL_MEM_WRITE_ONLY,
                     n * sizeof(float));

  g_kernel.setArg(0, buf_in);
  g_kernel.setArg(1, buf_out);
  g_kernel.setArg(2, static_cast<int>(n));

  cl::NDRange global(n);
  g_queue.enqueueNDRangeKernel(g_kernel, cl::NullRange, global, cl::NullRange);

  g_queue.finish();

  g_queue.enqueueReadBuffer(buf_out,
                            CL_TRUE,
                            0,
                            n * sizeof(float),
                            output.data());

  return output;
}
