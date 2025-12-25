#include "gelu_ocl.h"
#include <CL/cl.h>
#include <iostream>
#include <vector>

const char* geluKernelSource = R"(
__kernel void gelu_kernel(__global const float* input, 
                          __global float* output, 
                          const int size) {
    int id = get_global_id(0);
    if (id < size) {
        float x = input[id];
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float exp_val = exp(2.0f * inner);
        
        output[id] = 0.5f * x * (1.0f + (exp_val - 1.0f) / (exp_val + 1.0f));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
  if (input.empty()) return std::vector<float>();

  size_t size = input.size();
  size_t memory = size * sizeof(float);

  cl_platform_id platforms[8];
  cl_uint num_platforms;
  clGetPlatformIDs(8, platforms, &num_platforms);

  if (platform < 0 || platform >= (int)num_platforms) return std::vector<float>();

  cl_device_id device;
  clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

  const char* sources[] = { geluKernelSource };
  cl_program program = clCreateProgramWithSource(context, 1, sources, nullptr, nullptr);
  clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);

  cl_kernel kernel = clCreateKernel(program, "gelu_kernel", nullptr);

  cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    memory, (void*)input.data(), nullptr);
  cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, memory, nullptr, nullptr);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
  clSetKernelArg(kernel, 2, sizeof(int), &size);

  size_t local_size = 256;
  size_t global_size = (size + local_size - 1) / local_size * local_size;
  clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);

  std::vector<float> output(size);
  clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, memory, output.data(), 0, nullptr, nullptr);

  clReleaseMemObject(outputBuffer);
  clReleaseMemObject(inputBuffer);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return output;
}