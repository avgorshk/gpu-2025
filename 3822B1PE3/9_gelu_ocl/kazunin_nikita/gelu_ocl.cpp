#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

#include "gelu_ocl.h"
#include <CL/opencl.hpp>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct OCLContext {
  cl::Context ctx;
  cl::Device dev;
  cl::CommandQueue queue;
  cl::Kernel kernel;
  bool ready = false;
  std::mutex lock;
};

OCLContext global_ocl;

const char *kGeluKernel = R"CLC(
__kernel void gelu_kernel(__global const float* x,
                          __global float* y,
                          int n)
{
    int i = get_global_id(0);
    if (i >= n) return;

    float v = x[i];
    float v3 = v * v * v;

    // коэффициент sqrt(2/pi)
    const float K = 0.7978845608f;

    float u = K * (v + 0.044715f * v3);
    float e = exp(2.0f * u);

    float t = (e - 1.0f) / (e + 1.0f);  // tanh
    y[i] = 0.5f * v * (1.0f + t);
}
)CLC";

void initOnce(int platform_id) {
  std::lock_guard<std::mutex> guard(global_ocl.lock);
  if (global_ocl.ready)
    return;

  std::vector<cl::Platform> plats;
  cl::Platform::get(&plats);
  if (plats.empty())
    throw std::runtime_error("No OpenCL platforms");

  if (platform_id < 0 || platform_id >= (int)plats.size())
    platform_id = 0;

  cl::Platform plat = plats[platform_id];

  std::vector<cl::Device> devs;
  plat.getDevices(CL_DEVICE_TYPE_GPU, &devs);
  if (devs.empty())
    throw std::runtime_error("No GPU devices");

  global_ocl.dev = devs[0];
  global_ocl.ctx = cl::Context({global_ocl.dev});
  global_ocl.queue = cl::CommandQueue(global_ocl.ctx, global_ocl.dev);

  cl::Program prog(global_ocl.ctx, kGeluKernel);
  try {
    prog.build({global_ocl.dev});
  } catch (...) {
    std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(global_ocl.dev);
    std::cerr << "Build log:\n" << log << "\n";
    throw;
  }

  global_ocl.kernel = cl::Kernel(prog, "gelu_kernel");

  global_ocl.ready = true;
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float> &input, int platform_id) {
  if (input.empty())
    return {};

  initOnce(platform_id);

  size_t n = input.size();
  size_t bytes = n * sizeof(float);

  cl::Buffer buf_in(global_ocl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    bytes, (void *)input.data());
  cl::Buffer buf_out(global_ocl.ctx, CL_MEM_WRITE_ONLY, bytes);

  global_ocl.kernel.setArg(0, buf_in);
  global_ocl.kernel.setArg(1, buf_out);
  global_ocl.kernel.setArg(2, (int)n);

  global_ocl.queue.enqueueNDRangeKernel(global_ocl.kernel, cl::NullRange,
                                        cl::NDRange(n), cl::NullRange);
  global_ocl.queue.finish();

  std::vector<float> out(n);
  global_ocl.queue.enqueueReadBuffer(buf_out, CL_TRUE, 0, bytes, out.data());

  return out;
}
