#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <iostream>

const char* gelu_kernel_source = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    // Быстрая аппроксимация GELU: x * σ(1.702x)
    float sigmoid = 1.0f / (1.0f + exp(-1.702f * x));
    output[idx] = x * sigmoid;
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
	if (input.empty()) return {};

	cl_int err;

	cl_uint num_platforms;
	clGetPlatformIDs(0, NULL, &num_platforms);
	std::vector<cl_platform_id> platforms(num_platforms);
	clGetPlatformIDs(num_platforms, platforms.data(), NULL);

	if (platform >= (int)num_platforms) return {};

	cl_uint num_devices;
	clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	std::vector<cl_device_id> devices(num_devices);
	clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), NULL);

	if (num_devices == 0) return {};

	cl_context context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, &err);
	cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, &err);

	cl_program program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, NULL, &err);
	clBuildProgram(program, 1, &devices[0], NULL, NULL, NULL);
	cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);

	size_t n = input.size();
	size_t size = n * sizeof(float);

	cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		size, (void*)input.data(), &err);
	cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, &err);

	std::vector<float> output(n);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
	clSetKernelArg(kernel, 2, sizeof(int), &n);

	size_t global_size = (n + 255) / 256 * 256;
	size_t local_size = 256;

	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

	clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, size, output.data(), 0, NULL, NULL);

	clReleaseMemObject(d_input);
	clReleaseMemObject(d_output);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return output;
}