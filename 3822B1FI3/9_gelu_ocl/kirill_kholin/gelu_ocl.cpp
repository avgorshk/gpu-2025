#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

static const char *GELU_KERNEL = R"CLC(
__kernel void gelu_kernel(__global const float* input_data,
                          __global float* output_data,
                          const int total_elements)
{
    int idx = get_global_id(0);
    if (idx < total_elements) {
        float x = input_data[idx];
        float alpha = 1.702f;
        float sigmoid_input = alpha * x;
        float sigmoid_val = 1.0f / (1.0f + exp(-sigmoid_input));
        output_data[idx] = x * sigmoid_val;
    }
}
)CLC";

struct OpenCLState {
  cl_context context = nullptr;
  cl_device_id device = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
};

static std::unique_ptr<OpenCLState> global_state;

void InitOpenCL(int platform_idx) {
  if (global_state)
    return;

  global_state = std::make_unique<OpenCLState>();

  cl_uint platform_count;
  clGetPlatformIDs(0, NULL, &platform_count);
  if (platform_count == 0) {
    std::cerr << "No OpenCL platforms found" << std::endl;
    return;
  }

  std::vector<cl_platform_id> platforms(platform_count);
  clGetPlatformIDs(platform_count, platforms.data(), NULL);

  cl_uint device_count;
  clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_GPU, 0, NULL,
                 &device_count);
  if (device_count == 0) {
    std::cerr << "No GPU devices found" << std::endl;
    return;
  }

  std::vector<cl_device_id> devices(device_count);
  clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_GPU, device_count,
                 devices.data(), NULL);

  global_state->device = devices[0];
  global_state->context =
      clCreateContext(NULL, 1, &global_state->device, NULL, NULL, NULL);

  cl_queue_properties props[] = {CL_QUEUE_PROFILING_ENABLE, 0};
  global_state->queue = clCreateCommandQueueWithProperties(
      global_state->context, global_state->device, props, NULL);

  const char *source = GELU_KERNEL;
  size_t source_len = strlen(source);
  global_state->program = clCreateProgramWithSource(global_state->context, 1,
                                                    &source, &source_len, NULL);

  const char *options = "-cl-fast-relaxed-math -cl-mad-enable";
  clBuildProgram(global_state->program, 1, &global_state->device, options, NULL,
                 NULL);

  global_state->kernel =
      clCreateKernel(global_state->program, "gelu_kernel", NULL);
}

std::vector<float> GeluOCL(const std::vector<float> &input, int platform) {
  static bool initialized = false;
  if (!initialized) {
    InitOpenCL(platform);
    initialized = true;
  }

  if (!global_state || !global_state->kernel) {
    std::cerr << "OpenCL not initialized properly" << std::endl;
    return input;
  }

  size_t elem_count = input.size();
  size_t data_size = elem_count * sizeof(float);
  std::vector<float> output(elem_count);

  cl_int ret;
  cl_mem input_buf = clCreateBuffer(global_state->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    data_size, (void *)input.data(), &ret);
  cl_mem output_buf = clCreateBuffer(global_state->context, CL_MEM_WRITE_ONLY,
                                     data_size, NULL, &ret);

  int count = static_cast<int>(elem_count);
  clSetKernelArg(global_state->kernel, 0, sizeof(cl_mem), &input_buf);
  clSetKernelArg(global_state->kernel, 1, sizeof(cl_mem), &output_buf);
  clSetKernelArg(global_state->kernel, 2, sizeof(int), &count);

  size_t local_size = 256;
  size_t global_size =
      ((elem_count + local_size - 1) / local_size) * local_size;

  clEnqueueNDRangeKernel(global_state->queue, global_state->kernel, 1, NULL,
                         &global_size, &local_size, 0, NULL, NULL);
  clEnqueueReadBuffer(global_state->queue, output_buf, CL_TRUE, 0, data_size,
                      output.data(), 0, NULL, NULL);

  clReleaseMemObject(input_buf);
  clReleaseMemObject(output_buf);

  return output;
}