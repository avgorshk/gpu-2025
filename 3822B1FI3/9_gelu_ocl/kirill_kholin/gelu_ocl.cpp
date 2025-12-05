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

std::vector<float> GeluOCL(const std::vector<float> &input, int platform_id) {
  const int element_count = input.size();
  if (element_count == 0) {
    return std::vector<float>();
  }

  std::vector<float> output(element_count);

  cl_uint platform_count;
  clGetPlatformIDs(0, nullptr, &platform_count);
  cl_platform_id *platforms = new cl_platform_id[platform_count];
  clGetPlatformIDs(platform_count, platforms, nullptr);

  cl_platform_id target_platform = platforms[platform_id];
  delete[] platforms;

  cl_device_id device_id;
  cl_uint device_count;
  clGetDeviceIDs(target_platform, CL_DEVICE_TYPE_GPU, 1, &device_id,
                 &device_count);

  cl_int error;
  cl_context context =
      clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &error);
  cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &error);

  const size_t buffer_size = element_count * sizeof(float);
  cl_mem device_input =
      clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, nullptr, &error);
  cl_mem device_output =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer_size, nullptr, &error);

  clEnqueueWriteBuffer(queue, device_input, CL_TRUE, 0, buffer_size,
                       input.data(), 0, nullptr, nullptr);

  cl_program program =
      clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &error);
  clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);

  cl_kernel kernel = clCreateKernel(program, "gelu", &error);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_output);
  clSetKernelArg(kernel, 2, sizeof(int), &element_count);

  const size_t global_size = (element_count + 255) / 256 * 256;
  const size_t local_size = 256;
  clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size,
                         0, nullptr, nullptr);

  clEnqueueReadBuffer(queue, device_output, CL_TRUE, 0, buffer_size,
                      output.data(), 0, nullptr, nullptr);

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(device_input);
  clReleaseMemObject(device_output);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return output;
}