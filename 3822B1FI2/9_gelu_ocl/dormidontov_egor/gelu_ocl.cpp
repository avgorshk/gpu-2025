#include <CL/cl.h>

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gelu_ocl.h"

static const char* kGeluKernelSource = R"CLC(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;

    const float c = 0.044715f;
    const float sqrt_2_over_pi = 0.7978845608028654f;

    float x  = input[i];
    float x3 = x * x * x;
    float z  = sqrt_2_over_pi * (x + c * x3);

    
    float e  = exp(-2.0f * z);
    float t  = (1.0f - e) / (1.0f + e);


    float y = 0.5f * x * (1.0f + t);

    output[i] = y;
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input, int platformIndex) {
  const size_t n = input.size();
  std::vector<float> output(n);
  if (n == 0) {
    return output;
  }

  static bool initialized = false;
  static cl_context context = nullptr;
  static cl_command_queue queue = nullptr;
  static cl_program program = nullptr;
  static cl_kernel kernel = nullptr;
  static cl_mem d_input = nullptr;
  static cl_mem d_output = nullptr;
  static size_t capacity = 0;
  static cl_device_id device = nullptr;
  static int usedPlatformIndex = -1;

  cl_int err = CL_SUCCESS;

  if (!initialized) {
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
      throw std::runtime_error("clGetPlatformIDs failed");
    }

    if (platformIndex < 0 || (cl_uint)platformIndex >= numPlatforms) {
      throw std::runtime_error("Invalid platform index in GeluOCL");
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("clGetPlatformIDs (2) failed");
    }

    cl_platform_id platform = platforms[platformIndex];

    cl_uint numDevices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
      throw std::runtime_error(
          "No GPU devices found for given platform in GeluOCL");
    }

    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices,
                         devices.data(), nullptr);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("clGetDeviceIDs failed in GeluOCL");
    }

    device = devices[0];

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!context || err != CL_SUCCESS) {
      throw std::runtime_error("clCreateContext failed in GeluOCL");
    }

    queue = clCreateCommandQueue(context, device, 0, &err);
    if (!queue || err != CL_SUCCESS) {
      throw std::runtime_error("clCreateCommandQueue failed in GeluOCL");
    }

    const char* src = kGeluKernelSource;
    size_t srcLen = std::strlen(src);
    program = clCreateProgramWithSource(context, 1, &src, &srcLen, &err);
    if (!program || err != CL_SUCCESS) {
      throw std::runtime_error("clCreateProgramWithSource failed in GeluOCL");
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t logSize = 0;
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                            &logSize);
      std::vector<char> log(logSize);
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize,
                            log.data(), nullptr);
      std::cerr << "OpenCL build log:\n" << log.data() << std::endl;
      throw std::runtime_error("clBuildProgram failed in GeluOCL");
    }

    kernel = clCreateKernel(program, "gelu_kernel", &err);
    if (!kernel || err != CL_SUCCESS) {
      throw std::runtime_error("clCreateKernel failed in GeluOCL");
    }

    usedPlatformIndex = platformIndex;
    initialized = true;
  } else {
    (void)usedPlatformIndex;
  }

  if (n > capacity) {
    if (d_input) {
      clReleaseMemObject(d_input);
      d_input = nullptr;
    }
    if (d_output) {
      clReleaseMemObject(d_output);
      d_output = nullptr;
    }

    d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float),
                             nullptr, &err);
    if (!d_input || err != CL_SUCCESS) {
      capacity = 0;
      throw std::runtime_error("clCreateBuffer d_input failed in GeluOCL");
    }

    d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float),
                              nullptr, &err);
    if (!d_output || err != CL_SUCCESS) {
      clReleaseMemObject(d_input);
      d_input = nullptr;
      capacity = 0;
      throw std::runtime_error("clCreateBuffer d_output failed in GeluOCL");
    }

    capacity = n;
  }

  err = clEnqueueWriteBuffer(queue, d_input, CL_FALSE, 0, n * sizeof(float),
                             input.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("clEnqueueWriteBuffer failed in GeluOCL");
  }

  cl_int n_int = static_cast<cl_int>(n);
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_int), &n_int);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("clSetKernelArg failed in GeluOCL");
  }

  size_t globalWorkSize = ((n + 255) / 256) * 256;
  size_t localWorkSize = 256;

  err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize,
                               &localWorkSize, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("clEnqueueNDRangeKernel failed in GeluOCL");
  }

  err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, n * sizeof(float),
                            output.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("clEnqueueReadBuffer failed in GeluOCL");
  }

  clFinish(queue);

  return output;
}
