#include <CL/cl.h>

#include <cstring>
#include <string>
#include <vector>

#include "gelu_ocl.h"

static const char* kernelSrc = R"CLC(
__kernel void gelu_kernel(__global const float* in,
                          __global float* out,
                          const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;

    const float x = in[i];
    const float a = 0.7978845608028654f; // sqrt(2/pi)
    const float b = 0.044715f;

    const float x3 = x * x * x;
    const float z = a * (x + b * x3);
    const float e2 = native_exp(-2.0f * z);
    const float tanh_z = (1.0f - e2) / (1.0f + e2);
    out[i] = 0.5f * x * (1.0f + tanh_z);
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
  const int n = static_cast<int>(input.size());
  std::vector<float> output(n);

  cl_uint numPlatforms;
  clGetPlatformIDs(0, nullptr, &numPlatforms);
  std::vector<cl_platform_id> plats(numPlatforms);
  clGetPlatformIDs(numPlatforms, plats.data(), nullptr);
  cl_platform_id plat = plats[platform];

  cl_uint numDevices;
  clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
  std::vector<cl_device_id> devs(numDevices);
  clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, numDevices, devs.data(), nullptr);
  cl_device_id dev = devs[0];

  cl_int err;
  cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  cl_command_queue queue = clCreateCommandQueue(ctx, dev, 0, &err);

  const size_t srcSize = std::strlen(kernelSrc);
  cl_program program =
      clCreateProgramWithSource(ctx, 1, &kernelSrc, &srcSize, &err);
  clBuildProgram(program, 1, &dev, nullptr, nullptr, nullptr);

  cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);

  size_t bytes = sizeof(float) * n;
  cl_mem inBuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                bytes, (void*)input.data(), &err);
  cl_mem outBuf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inBuf);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &outBuf);
  clSetKernelArg(kernel, 2, sizeof(int), &n);

  size_t local = 256;
  size_t global = ((size_t)n + local - 1) / local * local;
  clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr,
                         nullptr);


  clEnqueueReadBuffer(queue, outBuf, CL_TRUE, 0, bytes, output.data(), 0,
                      nullptr, nullptr);

  clReleaseMemObject(inBuf);
  clReleaseMemObject(outBuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return output;
}
