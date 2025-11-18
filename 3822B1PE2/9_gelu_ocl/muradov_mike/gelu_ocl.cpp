#define CL_TARGET_OPENCL_VERSION 200
#include "gelu_ocl.h"
#include <CL/cl.h>

const char* KERNEL_SRC = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    float x3 = x * x * x;
    float arg = 0.7978845608028654f * (x + 0.044715f * x3);
    float e2x = exp(2.0f * arg);
    float tanh_val = (e2x - 1.0f) / (e2x + 1.0f);
    output[idx] = 0.5f * x * (1.0f + tanh_val);
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    
    cl_device_id device;
    clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, NULL);
    
    cl_program program = clCreateProgramWithSource(context, 1, &KERNEL_SRC, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", NULL);
    
    const int n = input.size();
    std::vector<float> output(n);
    
    cl_mem d_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  n * sizeof(float), (void*)input.data(), NULL);
    cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, NULL);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 2, sizeof(int), &n);
    
    size_t global_size = n;
    size_t local_size = 256;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, n * sizeof(float), output.data(), 0, NULL, NULL);
    
    clReleaseMemObject(d_out);
    clReleaseMemObject(d_in);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return output;
}
