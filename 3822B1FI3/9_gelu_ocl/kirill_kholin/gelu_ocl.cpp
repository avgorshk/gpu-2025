#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cstring>
#include <vector>

const char *GELU_KERNEL = R"(
__kernel void gelu(__global const float* input, __global float* output, const int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    float cdf = 0.5f * (1.0f + erf(x * 0.7071067811865475f));
    output[idx] = x * cdf;
}
)";

cl_context context = nullptr;
cl_device_id device = nullptr;
cl_command_queue queue = nullptr;
cl_program program = nullptr;
cl_kernel kernel = nullptr;

std::vector<float> GeluOCL(const std::vector<float> &input, int platform) {
  std::vector<float> results(input.size());

  if (!context) {
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, NULL, &numPlatforms);
    cl_platform_id platform_id = NULL;

    if (0 < numPlatforms) {
      cl_platform_id *platforms = new cl_platform_id[numPlatforms];
      clGetPlatformIDs(numPlatforms, platforms, NULL);
      platform_id = platforms[platform];
      delete[] platforms;
    }

    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};

    context = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, NULL,
                                      NULL, NULL);

    size_t size = 0;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);

    if (size > 0) {
      cl_device_id *devices = (cl_device_id *)alloca(size);
      clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
      device = devices[0];
    }

    queue = clCreateCommandQueue(context, device, 0, NULL);

    size_t source_len = strlen(GELU_KERNEL);
    program =
        clCreateProgramWithSource(context, 1, &GELU_KERNEL, &source_len, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "gelu", NULL);
  }

  size_t data_size = input.size() * sizeof(float);
  cl_mem input_buffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, NULL);
  cl_mem output_buffer =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, NULL, NULL);

  cl_event write_event;
  clEnqueueWriteBuffer(queue, input_buffer, CL_FALSE, 0, data_size,
                       input.data(), 0, NULL, &write_event);

  int count = static_cast<int>(input.size());
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
  clSetKernelArg(kernel, 2, sizeof(int), &count);

  size_t global_size = input.size();
  size_t local_size = 256;

  cl_event kernel_event;
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 1,
                         &write_event, &kernel_event);

  cl_event read_event;
  clEnqueueReadBuffer(queue, output_buffer, CL_FALSE, 0, data_size,
                      results.data(), 1, &kernel_event, &read_event);

  clWaitForEvents(1, &read_event);

  clReleaseMemObject(input_buffer);
  clReleaseMemObject(output_buffer);
  clReleaseEvent(write_event);
  clReleaseEvent(kernel_event);
  clReleaseEvent(read_event);

  return results;
}