#include "gelu_ocl.h"
#include <CL/opencl.h>
#include <string.h>

const char *source = R"(
__kernel void geluKernel (__global float *input, 
                      __global float *output,
                      const int count){

  int i = get_global_id(0);
  if (i < count) {
      float x = input[i];
      float x_cubed = x * x * x;
      float sqrt_2_pi = 0.7978845608028654f; // sqrt(2 / M_PI);
      output[i] = 0.5f * x * (1.0f + tanh(sqrt_2_pi * (x + 0.044715f * x_cubed)));
  }

})";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
  int n = input.size();
  std::vector<float> output(n);

  cl_uint numPlatforms = 0;

  clGetPlatformIDs(0, NULL, &numPlatforms);

  cl_platform_id _platform = NULL;

  if (0 < numPlatforms) {
    cl_platform_id * platforms = new cl_platform_id[numPlatforms];
    clGetPlatformIDs(numPlatforms, platforms, NULL);
    _platform = platforms[platform];
    delete[] platforms;
  }

  cl_context_properties properties[3] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties) _platform, 0
  };

  cl_context context = clCreateContextFromType (
    (NULL == _platform) ? NULL : properties,
    CL_DEVICE_TYPE_GPU, 
    NULL,
    NULL,
    NULL                                    );

  size_t size = 0;
  clGetContextInfo (
    context,
    CL_CONTEXT_DEVICES,
    0,
    NULL,
    &size       );

  cl_device_id device;

  if (size > 0) {
    cl_device_id * devices = ( cl_device_id * ) alloca (size);
    clGetContextInfo (
      context,
      CL_CONTEXT_DEVICES,
      size,
      devices,
      NULL   );
    device = devices[0];
  }

  cl_command_queue queue = clCreateCommandQueue(
    context,
    device, 
    0,
    NULL
  );

  size_t srclen[] = { strlen(source) };
  
  cl_program program = clCreateProgramWithSource(
    context,
    1,
    &source,
    srclen,
    NULL
  );

  clBuildProgram(
    program, 1, &device,
    NULL, NULL, NULL);

  cl_kernel kernel = clCreateKernel( program, "geluKernel", NULL);

  cl_mem data = clCreateBuffer(
    context,
    CL_MEM_READ_ONLY,
    sizeof(float) * n,
    NULL,
    NULL                    );

  cl_mem results = clCreateBuffer(
    context,
    CL_MEM_WRITE_ONLY,
    sizeof(float) * n,
    NULL,
    NULL                    );

  clEnqueueWriteBuffer(
    queue,
    data,
    CL_TRUE,
    0,
    sizeof(float) * n,
    input.data(),
    0,
    NULL,
    NULL              );

  const size_t count = n;

  clSetKernelArg(
    kernel,
    0,
    sizeof(cl_mem),
    &data       );
  
  clSetKernelArg(
    kernel,
    1,
    sizeof(cl_mem),
    &results       );

    
  clSetKernelArg(
    kernel,
    2,
    sizeof(int),
    &n       );

  size_t group;

  clGetKernelWorkGroupInfo(
    kernel,
    device,
    CL_KERNEL_WORK_GROUP_SIZE,
    sizeof(size_t),
    &group,
    NULL
  );

  clEnqueueNDRangeKernel(
    queue,
    kernel,
    1,
    NULL,
    &count,
    &group,
    0,
    NULL,
    NULL
  );

  clFinish(queue);

  clEnqueueReadBuffer(
    queue,
    results,
    CL_TRUE,
    0,
    sizeof(float) * count,
    output.data(),
    0,
    NULL,
    NULL
  );

  clReleaseMemObject(data);
  clReleaseMemObject(results);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  
  return output;
}
