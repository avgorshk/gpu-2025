#include "gelu_ocl.h"
#include <CL/cl.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>


static const char *gelu_kernel_source = R"CL_KERNEL(
__kernel void gelu(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    float x = input[idx];
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float z = sqrt_2_over_pi * (x + coeff * x3);
    output[idx] = 0.5f * x * (1.0f + tanh(z));
}
)CL_KERNEL";

std::vector<float> GeluOCL(const std::vector<float> &input, int platform_idx) {
  if (input.empty()) {
    return std::vector<float>();
  }

  cl_uint num_platforms = 0;
  cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    throw std::runtime_error("No OpenCL platforms found");
  }

  if (platform_idx < 0 || static_cast<cl_uint>(platform_idx) >= num_platforms) {
    throw std::invalid_argument("Invalid platform index");
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to get OpenCL platforms");
  }

  cl_platform_id platform = platforms[platform_idx];

  cl_uint num_devices = 0;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  if (err != CL_SUCCESS || num_devices == 0) {
    throw std::runtime_error("No GPU devices found on selected platform");
  }

  std::vector<cl_device_id> devices(num_devices);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices,
                       devices.data(), nullptr);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to get GPU devices");
  }

  cl_device_id device = devices[0];

  cl_context context =
      clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create OpenCL context");
  }

  cl_command_queue queue =
      clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  if (err != CL_SUCCESS) {
    clReleaseContext(context);
    throw std::runtime_error("Failed to create OpenCL command queue");
  }

  const char *source = gelu_kernel_source;
  size_t source_len = std::strlen(source);
  cl_program program =
      clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  if (err != CL_SUCCESS) {
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    throw std::runtime_error("Failed to create OpenCL program");
  }

  err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &log_size);
    if (log_size > 1) {
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), nullptr);
    }
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    throw std::runtime_error("Failed to build OpenCL program");
  }

  cl_kernel kernel = clCreateKernel(program, "gelu", &err);
  if (err != CL_SUCCESS) {
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    throw std::runtime_error("Failed to create OpenCL kernel");
  }

  size_t n = input.size();
  size_t bytes = n * sizeof(float);

  cl_mem input_buf =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                     (void *)input.data(), &err);
  if (err != CL_SUCCESS) {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    throw std::runtime_error("Failed to create input buffer");
  }

  cl_mem output_buf =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    throw std::runtime_error("Failed to create output buffer");
  }

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_int), &n);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(output_buf);
    clReleaseMemObject(input_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    throw std::runtime_error("Failed to set kernel arguments");
  }

  size_t global_work_size = ((n + 255) / 256) * 256;
  size_t local_work_size = 256;

  err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size,
                               &local_work_size, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(output_buf);
    clReleaseMemObject(input_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    throw std::runtime_error("Failed to enqueue kernel");
  }

  std::vector<float> output(n);
  err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, bytes, output.data(),
                            0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(output_buf);
    clReleaseMemObject(input_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    throw std::runtime_error("Failed to read output buffer");
  }

  clReleaseMemObject(output_buf);
  clReleaseMemObject(input_buf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return output;
}