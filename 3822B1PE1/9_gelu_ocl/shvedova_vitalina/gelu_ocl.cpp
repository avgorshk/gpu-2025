#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <cassert>

namespace {
const char* kKernelSrc = R"(
__kernel void gelu(__global const float* input, __global float* output, int n) {
  int i = get_global_id(0);
  if (i >= n) return;
  float x = input[i];
  float inner = 0.797884560802f * (x + 0.044715f * x * x * x);
  float e = exp(2.0f * inner);
  float t = (e - 1.0f) / (e + 1.0f);
  output[i] = 0.5f * x * (1.0f + t);
}
)";

cl::Context g_ctx;
cl::CommandQueue g_q;
cl::Kernel g_k;
bool ready = false;

void Setup(int platform_id) {
  if (ready) return;
  std::vector<cl::Platform> plats;
  cl::Platform::get(&plats);
  if (plats.empty()) throw std::runtime_error("No OpenCL platforms found");
  if (platform_id >= (int)plats.size()) platform_id = 0;
  std::vector<cl::Device> devs;
  plats[platform_id].getDevices(CL_DEVICE_TYPE_GPU, &devs);
  assert(!devs.empty());
  g_ctx = cl::Context(devs);
  g_q = cl::CommandQueue(g_ctx, devs[0]);
  cl::Program prog(g_ctx, kKernelSrc);
  prog.build(devs);
  g_k = cl::Kernel(prog, "gelu");
  ready = true;
}
} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
  if (input.empty()) return {};
  Setup(platform);
  size_t n = input.size();

  cl::Buffer in(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                n * sizeof(float), (void*)input.data());
  cl::Buffer out(g_ctx, CL_MEM_WRITE_ONLY, n * sizeof(float));

  g_k.setArg(0, in);
  g_k.setArg(1, out);
  g_k.setArg(2, static_cast<int>(n));

  g_q.enqueueNDRangeKernel(g_k, cl::NullRange, cl::NDRange(n), cl::NullRange);
  std::vector<float> res(n);
  g_q.enqueueReadBuffer(out, CL_TRUE, 0, n * sizeof(float), res.data());
  return res;
}
