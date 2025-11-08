#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <vector>

const char kernel_src[] = R"(
__kernel void gelu(__global const float* __restrict__ input, __global float* __restrict__ output, int n) {
  const int idx = get_global_id(0);
  if (idx < n) {
    const float x = input[idx];
    output[idx] = x / (1.0f + exp(-2.0f * 0.797884560802f * (x + 0.044715f * (x * x * x))));
  }
}
)";

cl::Context ctx;
cl::CommandQueue queue;
cl::Kernel kernel;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
  static int initialized_platform = -1;

  if (initialized_platform != platform) {
    initialized_platform = platform;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform pl = platforms[platform];

    std::vector<cl::Device> devices;
    pl.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    ctx = cl::Context(devices);
    queue = cl::CommandQueue(ctx, devices[0]);

    cl::Program program(ctx, kernel_src);
    program.build(devices);
    kernel = cl::Kernel(program, "gelu");
  }

  cl::Buffer input_buf(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float), const_cast<float*>(input.data()));
  cl::Buffer output_buf(ctx, CL_MEM_WRITE_ONLY, input.size() * sizeof(float));

  kernel.setArg(0, input_buf);
  kernel.setArg(1, output_buf);
  kernel.setArg(2, static_cast<int>(input.size()));

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.size()), cl::NullRange);

  std::vector<float> output(input.size());

  queue.finish();

  queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, input.size() * sizeof(float), output.data());

  return output;
}
