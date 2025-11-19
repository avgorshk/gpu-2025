// Copyright (c) 2025 Ionova-Ekaterina
#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <iostream>

#define CHECK_CL_ERROR(callable)                                          \
  {                                                                       \
    auto codeError = callable;                                            \
    if (codeError != CL_SUCCESS) {                                        \
      std::cerr << "\033[1;31merror\033[0m: ";                            \
      std::cerr << "code error: " << static_cast<int>(codeError) << '\n'; \
      std::cerr << "loc: " << __FILE__ << '(' << __LINE__ << ")\n";       \
      std::exit(static_cast<int>(codeError));                             \
    }                                                                     \
  }

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);

  if (all_platforms.empty()) {
    std::cerr << "\033[1;31merror\033[0m: no OpenCL platforms available\n";
    return {};
  }

  if (platform < 0 || platform >= static_cast<int>(all_platforms.size())) {
    std::cerr << "\033[1;31merror\033[0m: invalid platform index: " << platform << '\n';
    return {};
  }

  cl::Platform selected_platform = all_platforms[platform];

  std::vector<cl::Device> devices;
  selected_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

  if (devices.empty()) {
    std::cerr << "\033[1;31merror\033[0m: no GPU devices available on selected platform\n";
    return {};
  }

  cl::Device device = devices[0];

  cl::Context context(device);
  cl::CommandQueue queue(context);

  std::string codeKernel = R"(
__kernel void gelu_kernel(__global const float* x, __global float* y, int countElem) {
  int i = get_global_id(0);

  if (i < countElem) {
      float val = x[i];
      float x3 = val * val * val;
      float inner = val + 0.044715f * x3;
      float a = 0.7978845608f * inner;
      float exp_2a = exp(2.0f * a);
      float tanh_a = (exp_2a - 1.0f) / (exp_2a + 1.0f);
      y[i] = val * 0.5f * (1.0f + tanh_a);
  }
}
)";

  cl::Program::Sources sources;
  sources.emplace_back(std::move(codeKernel));

  cl::Program program(context, sources);
  if (program.build(devices) != CL_SUCCESS) {
    std::cerr << "\033[1;31merror\033[0m: ";
    std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << '\n';
    return {};
  }

  if (input.empty()) {
    return {};
  }

  auto size = input.size();
  auto countBytes = size * sizeof(float);

  cl::Buffer bufferInput(context, CL_MEM_READ_ONLY, countBytes);
  cl::Buffer bufferOutput(context, CL_MEM_WRITE_ONLY, countBytes);

  CHECK_CL_ERROR(queue.enqueueWriteBuffer(bufferInput, CL_FALSE, 0, countBytes, input.data()));

  cl::Kernel kernel(program, "gelu_kernel");
  kernel.setArg(0, bufferInput);
  kernel.setArg(1, bufferOutput);
  kernel.setArg(2, static_cast<int>(size));

  CHECK_CL_ERROR(queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange));

  std::vector<float> output(size);
  CHECK_CL_ERROR(queue.enqueueReadBuffer(bufferOutput, CL_FALSE, 0, countBytes, output.data()));

  CHECK_CL_ERROR(queue.finish());

  return output;
}