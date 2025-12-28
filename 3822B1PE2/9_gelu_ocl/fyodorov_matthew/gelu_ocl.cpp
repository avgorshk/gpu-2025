#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

const char *gelu_kernel_source = R"CLC(
__kernel void gelu_kernel(__global const float* input, 
                          __global float* output, 
                          const unsigned int size) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        float x = input[idx];
        
        const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/π)
        const float coeff = 0.044715f;
        
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        float exp_val = exp(-2.0f * fabs(inner));
        float tanh_val = copysign(1.0f - exp_val, inner) / (1.0f + exp_val);
        
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}
)CLC";

struct OpenCLContext {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;

  OpenCLContext()
      : platform(0), device(0), context(0), queue(0), program(0), kernel(0) {}

  ~OpenCLContext() { cleanup(); }

  void cleanup() {
    if (kernel)
      clReleaseKernel(kernel);
    if (program)
      clReleaseProgram(program);
    if (queue)
      clReleaseCommandQueue(queue);
    if (context)
      clReleaseContext(context);
  }
};

static OpenCLContext *g_context = nullptr;

static void initOpenCLContext(int platform_idx) {
  if (g_context) {
    delete g_context;
  }

  g_context = new OpenCLContext();
  cl_int err;

  cl_uint num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    throw std::runtime_error("Error");
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Error");
  }

  if (platform_idx >= static_cast<int>(num_platforms)) {
    throw std::runtime_error("Error");
  }
  g_context->platform = platforms[platform_idx];

  cl_uint num_devices;
  err = clGetDeviceIDs(g_context->platform, CL_DEVICE_TYPE_GPU, 0, NULL,
                       &num_devices);

  if (err != CL_SUCCESS || num_devices == 0) {
    err = clGetDeviceIDs(g_context->platform, CL_DEVICE_TYPE_CPU, 0, NULL,
                         &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
      throw std::runtime_error("Error");
    }
  }

  std::vector<cl_device_id> devices(num_devices);
  err = clGetDeviceIDs(g_context->platform, CL_DEVICE_TYPE_GPU, num_devices,
                       devices.data(), NULL);

  if (err != CL_SUCCESS) {
    err = clGetDeviceIDs(g_context->platform, CL_DEVICE_TYPE_CPU, num_devices,
                         devices.data(), NULL);
  }

  if (err != CL_SUCCESS) {
    throw std::runtime_error("Error");
  }

  g_context->device = devices[0];

  char device_name[128];
  clGetDeviceInfo(g_context->device, CL_DEVICE_NAME, sizeof(device_name),
                  device_name, NULL);
  std::cout << "Error" << device_name << std::endl;

  g_context->context =
      clCreateContext(NULL, 1, &g_context->device, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Error");
  }

  g_context->queue = clCreateCommandQueue(g_context->context, g_context->device,
                                          CL_QUEUE_PROFILING_ENABLE, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Error");
  }

  const char *sources[] = {gelu_kernel_source};
  g_context->program =
      clCreateProgramWithSource(g_context->context, 1, sources, NULL, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Error");
  }

  err = clBuildProgram(g_context->program, 1, &g_context->device, NULL, NULL,
                       NULL);

  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(g_context->program, g_context->device,
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(g_context->program, g_context->device,
                          CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);

    std::string error_msg = "Error";
    error_msg += std::string(log.data(), log.size());
    throw std::runtime_error(error_msg);
  }

  g_context->kernel = clCreateKernel(g_context->program, "gelu_kernel", &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Error");
  }
}

std::vector<float> GeluOCL(const std::vector<float> &input, int platform) {
  static bool initialized = false;
  if (!initialized) {
    try {
      initOpenCLContext(platform);
      initialized = true;
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("Error") + e.what());
    }
  }

  if (!g_context) {
    throw std::runtime_error("Error");
  }

  size_t size = input.size();
  std::vector<float> output(size, 0.0f);

  cl_int err;

  cl_mem d_input = clCreateBuffer(
      g_context->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      size * sizeof(float), const_cast<float *>(input.data()), &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Error");
  }

  cl_mem d_output = clCreateBuffer(g_context->context, CL_MEM_WRITE_ONLY,
                                   size * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(d_input);
    throw std::runtime_error("Error");
  }

  err = clSetKernelArg(g_context->kernel, 0, sizeof(cl_mem), &d_input);
  err |= clSetKernelArg(g_context->kernel, 1, sizeof(cl_mem), &d_output);
  err |= clSetKernelArg(g_context->kernel, 2, sizeof(unsigned int), &size);

  if (err != CL_SUCCESS) {
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    throw std::runtime_error("Error");
  }

  size_t local_work_size = 256;
  size_t global_work_size =
      ((size + local_work_size - 1) / local_work_size) * local_work_size;

  err = clEnqueueNDRangeKernel(g_context->queue, g_context->kernel, 1, NULL,
                               &global_work_size, &local_work_size, 0, NULL,
                               NULL);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    throw std::runtime_error("Error");
  }

  err = clEnqueueReadBuffer(g_context->queue, d_output, CL_TRUE, 0,
                            size * sizeof(float), output.data(), 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    throw std::runtime_error("Error");
  }

  clFinish(g_context->queue);

  clReleaseMemObject(d_input);
  clReleaseMemObject(d_output);

  return output;
}

std::vector<std::string> getOpenCLPlatforms() {
  cl_uint num_platforms;
  cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);

  if (err != CL_SUCCESS || num_platforms == 0) {
    return {};
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);

  std::vector<std::string> platform_names;

  for (size_t i = 0; i < platforms.size(); ++i) {
    char name[256];
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
    platform_names.push_back(std::string(name) + " (ID: " + std::to_string(i) +
                             ")");
  }

  return platform_names;
}