#include "gelu_ocl.h"
#include <CL/cl.hpp>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>


static const char *gelu_kernel_source = R"CL_KERNEL(
__kernel void gelu(__global const float* input, __global float* output, const int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    // Optimized GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    // Using tanh(z) = (exp(2z) - 1) / (exp(2z) + 1) for better performance
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/π)
    const float coeff = 0.044715f;
    
    float x3 = x * x * x;
    float z = sqrt_2_over_pi * (x + coeff * x3);
    float exp_2z = exp(2.0f * z);
    float tanh_z = (exp_2z - 1.0f) / (exp_2z + 1.0f);
    
    output[idx] = 0.5f * x * (1.0f + tanh_z);
}
)CL_KERNEL";

std::vector<float> GeluOCL(const std::vector<float> &input, int platform) {
  if (input.empty()) {
    return std::vector<float>();
  }

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (platform < 0 || static_cast<size_t>(platform) >= platforms.size()) {
    throw std::invalid_argument("Invalid platform index");
  }

  cl::Platform selected_platform = platforms[platform];

  std::vector<cl::Device> devices;
  selected_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

  if (devices.empty()) {
    throw std::runtime_error("No GPU devices found");
  }

  cl::Device device = devices[0];

  cl::Context context({device});
  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  cl::Program program(context, gelu_kernel_source);
  program.build({device});
  cl::Kernel kernel(program, "gelu");

  size_t n = input.size();
  size_t bytes = n * sizeof(float);

  cl::Buffer input_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                       (void *)input.data());
  cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY, bytes);

  kernel.setArg(0, input_buf);
  kernel.setArg(1, output_buf);
  kernel.setArg(2, static_cast<int>(n));

  size_t global_work_size = ((n + 255) / 256) * 256;
  size_t local_work_size = 256;

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size,
                             local_work_size);

  std::vector<float> output(n);
  queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, bytes, output.data());

  return output;
}