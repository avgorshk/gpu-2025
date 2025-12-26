#include "gelu_ocl.h"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <iostream>
#include <string>
#include <vector>


// OpenCL Kernel Source
const char *ocl_source = R"(
__kernel void gelu(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = in[i];
        // Tanh approximation: 0.5x(1 + tanh(sqrt(2/pi)*(x + 0.044715x^3)))
        const float SQRT_2_OVER_PI = 0.7978845608f;
        const float COEF = 0.044715f;
        
        float val = SQRT_2_OVER_PI * (x + COEF * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanh(val));
    }
}
)";

// Helper to check errors
#define CHECK_CL(err)                                                          \
  if (err != CL_SUCCESS) {                                                     \
    std::cerr << "OpenCL Error: " << err << " at line " << __LINE__            \
              << std::endl;                                                    \
    exit(1);                                                                   \
  }

std::vector<float> GeluOCL(const std::vector<float> &input, int platform_id) {
  cl_int err;
  int n = static_cast<int>(input.size());
  std::vector<float> output(n);

  // 1. Get Platform
  cl_uint num_platforms;
  CHECK_CL(clGetPlatformIDs(0, nullptr, &num_platforms));
  if (num_platforms == 0) {
    std::cerr << "No OpenCL platforms found." << std::endl;
    return output;
  }
  std::vector<cl_platform_id> platforms(num_platforms);
  CHECK_CL(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

  // Select requested platform or default to 0
  cl_platform_id platform = platforms[0];
  if (platform_id >= 0 && platform_id < (int)num_platforms) {
    platform = platforms[platform_id];
  }

  // 2. Get Device (GPU)
  cl_device_id device;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
  if (err != CL_SUCCESS) {
    // Fallback to CPU if no GPU
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
  }
  CHECK_CL(err);

  // 3. Create Context
  cl_context context =
      clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  CHECK_CL(err);

  // 4. Create Command Queue
  // OpenCL 2.0+ uses clCreateCommandQueueWithProperties, older use
  // clCreateCommandQueue Using simple version for compatibility if possible, or
  // properties for modern
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_CL(err);

  // 5. Create Program and Kernel
  cl_program program =
      clCreateProgramWithSource(context, 1, &ocl_source, nullptr, &err);
  CHECK_CL(err);

  err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    // Print build log
    size_t len;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &len);
    std::vector<char> log(len);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len,
                          log.data(), nullptr);
    std::cerr << "Build Log: " << log.data() << std::endl;
    exit(1);
  }

  cl_kernel kernel = clCreateKernel(program, "gelu", &err);
  CHECK_CL(err);

  // 6. Allocate Memory
  size_t size_bytes = n * sizeof(float);
  cl_mem d_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               size_bytes, (void *)input.data(), &err);
  CHECK_CL(err);

  cl_mem d_out =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_bytes, nullptr, &err);
  CHECK_CL(err);

  // 7. Set Arguments
  CHECK_CL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
  CHECK_CL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
  CHECK_CL(clSetKernelArg(kernel, 2, sizeof(int), &n));

  // 8. Execute
  size_t global_work_size = n;
  // Round up to multiple of generic local size if needed, but OpenCL handles
  // NULL local_work_size automatically
  CHECK_CL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size,
                                  nullptr, 0, nullptr, nullptr));

  // 9. Read Result
  CHECK_CL(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, size_bytes,
                               output.data(), 0, nullptr, nullptr));

  // 10. Cleanup
  clReleaseMemObject(d_in);
  clReleaseMemObject(d_out);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return output;
}
