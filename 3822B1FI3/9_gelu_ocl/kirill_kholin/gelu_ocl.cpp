#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cmath>
#include <vector>

const char *kernelSource = R"CLC(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int total_elements)
{
    int element_index = get_global_id(0);
    
    float x = input[element_index];
    
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    
    float tanh_val = tanh(inner);
    
    output[element_index] = 0.5f * x * (1.0f + tanh_val);
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float> &input,
                           int platform_index) {
  cl_int error_code;
  cl_uint platform_count;

  clGetPlatformIDs(0, nullptr, &platform_count);
  std::vector<cl_platform_id> platforms(platform_count);
  clGetPlatformIDs(platform_count, platforms.data(), nullptr);

  cl_uint device_count;
  clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 0, nullptr,
                 &device_count);
  std::vector<cl_device_id> devices(device_count);
  clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, device_count,
                 devices.data(), nullptr);

  cl_context context =
      clCreateContext(nullptr, 1, &devices[0], nullptr, nullptr, &error_code);
  cl_command_queue_properties props = 0;
  cl_command_queue queue = clCreateCommandQueueWithProperties(
      context, devices[0], &props, &error_code);

  cl_program program = clCreateProgramWithSource(context, 1, &kernelSource,
                                                 nullptr, &error_code);
  clBuildProgram(program, 1, &devices[0], nullptr, nullptr, nullptr);

  cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &error_code);

  const size_t data_size = input.size();
  const size_t byte_size = data_size * sizeof(float);

  cl_mem input_buffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     byte_size, (void *)input.data(), &error_code);
  cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, byte_size,
                                        nullptr, &error_code);

  int element_count = static_cast<int>(data_size);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
  clSetKernelArg(kernel, 2, sizeof(int), &element_count);

  size_t global_work_size = data_size;
  size_t local_work_size = 256;

  clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size,
                         &local_work_size, 0, nullptr, nullptr);

  std::vector<float> result(data_size);
  clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, byte_size,
                      result.data(), 0, nullptr, nullptr);

  clReleaseMemObject(input_buffer);
  clReleaseMemObject(output_buffer);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return result;
}