#include "gelu_ocl.h"

#include <CL/cl.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

const char* GELU_kernel = R"CLC(
__kernel void gelu_kernel(__global const float* input, __global float* output, const int size) {
    int i = get_global_id(0);
    if (i < size) {
        float x = input[i];
        float c = 0.044715f;
        float sqrt_2_over_pi = 0.797884; // sqrt(2/pi)
        float x_cube = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + c * x_cube);
        output[i] = 0.5f * x * (1.0f + tanh(tanh_arg));
    }
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
  cl_int err;
  std::vector<float> output(input.size());

  cl_uint numPlatforms;
  err = clGetPlatformIDs(0, nullptr, &numPlatforms);
  if (err != CL_SUCCESS) {
    std::cerr << "Ошибка получения GPU устройства: " << err << std::endl;
    return {};
  }
  std::vector<cl_platform_id> platforms(numPlatforms);
  err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Ошибка получения GPU устройства: " << err << std::endl;
    return {};
  }
  cl_platform_id platform_id = platforms[platform];

  cl_uint numDevices;
  cl_device_id device;
  err =
      clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
  if (err != CL_SUCCESS) {
    std::cerr << "Ошибка получения GPU устройства: " << err << std::endl;
    return {};
  }
  cl_context context =
      clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  cl_queue_properties props[] = {0};
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, props, &err);

  cl_mem input_buffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * input.size(), (void*)input.data(), &err);
  cl_mem output_buffer = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY, sizeof(float) * output.size(), nullptr, &err);

  cl_program program =
      clCreateProgramWithSource(context, 1, &GELU_kernel, nullptr, &err);
  err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                          log.data(), nullptr);
    std::cerr << "Ошибка компиляции kernel: " << log.data() << std::endl;
    return {};
  }
  cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
  int size = static_cast<int>(input.size());
  clSetKernelArg(kernel, 2, sizeof(int), &size);

  size_t global_work_size = input.size();
  clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr,
                         0, nullptr, nullptr);

  clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
                      sizeof(float) * output.size(), output.data(), 0, nullptr,
                      nullptr);

  clReleaseMemObject(input_buffer);
  clReleaseMemObject(output_buffer);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return output;
}
