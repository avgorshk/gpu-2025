#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <cmath>

const char* GELU_KERNEL = R"(
__kernel void gelu(__global const float* input, __global float* output, const int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    output[idx] = 0.5f * x * (1.0f + tanh(inner));
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    
    cl_platform_id selected_platform = platforms[platform];
    
    cl_device_id device;
    clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
    
    const char* sources[] = { GELU_KERNEL };
    cl_program program = clCreateProgramWithSource(context, 1, sources, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(program, "gelu", NULL);
    
    size_t n = input.size();
    std::vector<float> output(n);
    
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                        n * sizeof(float), (void*)input.data(), NULL);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, NULL);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &n);
    
    size_t global_size = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, n * sizeof(float), output.data(), 0, NULL, NULL);
    
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(input_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return output;
}