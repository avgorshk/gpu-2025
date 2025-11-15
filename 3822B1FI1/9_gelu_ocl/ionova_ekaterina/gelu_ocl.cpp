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

std::vector<float> GeluOCL(const std::vector<float>& input) {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty()) {
    std::cerr << "\033[1;31merror\033[0m: no platform available\n";
    return {};
  }
  cl::Platform platform = platforms.front();

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  if (devices.empty()) {
    std::cerr << "\033[1;31merror\033[0m: no device available\n";
    return {};
  }
  cl::Device device = devices.front();

  cl::Context context(device);
  cl::CommandQueue queue(context);

  std::string codeKernel = R"(
__kernel void gelu_kernel(__global const float* x, __global float* y, int countElem) {
  int i = get_global_id(0);

  if (i < countElem) {
      float val = x[i];
      float tmp = val * (1.595769122f + val * val * 0.071354816f);
      y[i] = val - val / (1.0f + exp(tmp));
  }
}
)";

  cl::Program::Sources sources;
  sources.emplace_back(std::move(codeKernel));

  cl::Program program(context, sources);
  if (program.build() != CL_SUCCESS) {
    std::cerr << "\033[1;31merror\033[0m: ";
    std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << '\n';
    return {};
  }

  if (input.empty()) return {};
  auto size = input.size();
  auto countBytes = size * sizeof(float);

  cl::Buffer bufferInput(context, CL_MEM_READ_ONLY, countBytes);
  cl::Buffer bufferOutput(context, CL_MEM_WRITE_ONLY, countBytes);

  CHECK_CL_ERROR(queue.enqueueWriteBuffer(bufferInput, CL_TRUE, 0, countBytes,
                                          input.data()));

  cl::Kernel kernel(program, "gelu_kernel");
  kernel.setArg(0, bufferInput);
  kernel.setArg(1, bufferOutput);
  kernel.setArg(2, static_cast<int>(size));
  CHECK_CL_ERROR(queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(size), cl::NullRange));

  std::vector<float> output(size);
  CHECK_CL_ERROR(queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, countBytes,
                                         output.data()));

  return output;
}