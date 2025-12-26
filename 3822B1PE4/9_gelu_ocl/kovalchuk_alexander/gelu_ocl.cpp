#include "gelu_ocl.h"
#include <CL/opencl.h>
#include <cmath>
#include <string>

const char *kernelSource = R"(
__kernel void gelu(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        float x = input[idx];
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float exp_2x = exp(2.0f * inner);
        float tanh_val = (exp_2x - 1.0f) / (exp_2x + 1.0f);
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float> &input, int platform) {
  int n = input.size();
  if (n == 0) {
    return std::vector<float>();
  }

  std::vector<float> output(n);

  cl_platform_id platformId;
  cl_device_id deviceId;
  cl_uint numPlatforms, numDevices;

  clGetPlatformIDs(0, nullptr, &numPlatforms);
  cl_platform_id *platforms = new cl_platform_id[numPlatforms];
  clGetPlatformIDs(numPlatforms, platforms, nullptr);
  platformId = platforms[platform];
  delete[] platforms;

  clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, &numDevices);

  cl_int err;
  cl_context context =
      clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &err);
  cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, &err);

  size_t bytes = n * sizeof(float);
  cl_mem d_input =
      clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
  cl_mem d_output =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

  clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, bytes, input.data(), 0,
                       nullptr, nullptr);

  cl_program program =
      clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
  clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);

  cl_kernel kernel = clCreateKernel(program, "gelu", &err);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
  clSetKernelArg(kernel, 2, sizeof(int), &n);

  size_t globalSize = (n + 255) / 256 * 256;
  size_t localSize = 256;
  clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0,
                         nullptr, nullptr);

  clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, bytes, output.data(), 0,
                      nullptr, nullptr);

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(d_input);
  clReleaseMemObject(d_output);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return output;
}
