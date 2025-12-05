#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>

const char* GELU_KERNEL = R"(
__kernel void gelu_kernel(__global const float* in, __global float* out, const int n) {
    const float a = 0.79788456f;  // sqrt(2/pi)
    const float b = 0.044715f;
    int i = get_global_id(0);
    if (i < n) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + tanh(a * (x + b * x*x*x)));
    }
}
)";

cl_device_id find_gpu_device(cl_platform_id platform) {
  cl_uint num_devices = 0;
  cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  if (err != CL_SUCCESS || num_devices == 0) return nullptr;

  std::vector<cl_device_id> devices(num_devices);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
  if (err != CL_SUCCESS) return nullptr;

  return devices[0];
}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform_index) {
  if (input.empty()) return {};

  cl_int err;
  cl_uint num_platforms;
  err = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) throw std::runtime_error("No OpenCL platforms");

  std::vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (platform_index >= static_cast<int>(num_platforms)) throw std::runtime_error("Platform index out of range");

  cl_platform_id platform = platforms[platform_index];
  cl_device_id device = find_gpu_device(platform);
  if (!device) throw std::runtime_error("No GPU device on selected platform");

  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) throw std::runtime_error("Failed to create context");

  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  if (err != CL_SUCCESS) { clReleaseContext(context); throw std::runtime_error("Failed to create queue"); }

  cl_program program = clCreateProgramWithSource(context, 1, &GELU_KERNEL, nullptr, &err);
  if (err != CL_SUCCESS) throw std::runtime_error("Failed to create program");

  err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    std::string msg = "Build failed: ";
    msg += std::string(log.data());
    throw std::runtime_error(msg);
  }

  cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
  if (err != CL_SUCCESS) throw std::runtime_error("Failed to create kernel");

  size_t n = input.size();
  cl_mem buf_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), (void*)input.data(), &err);
  cl_mem buf_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &err);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
  clSetKernelArg(kernel, 2, sizeof(int), &n);

  size_t global = n;
  clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
  clFinish(queue);

  std::vector<float> output(n);
  clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, n * sizeof(float), output.data(), 0, nullptr, nullptr);

  clReleaseMemObject(buf_in);
  clReleaseMemObject(buf_out);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return output;
}
