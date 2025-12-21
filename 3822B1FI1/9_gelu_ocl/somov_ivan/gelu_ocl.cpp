#include "gelu_ocl.h"

#include <CL/cl.h>
#include <iostream>
#include <cmath>

const char* geluKernelSource = R"(
__kernel void gelu(__global const float* input, __global float* output, const int size) {
    int id = get_global_id(0);
    if (id < size) {
        float x = input[id];
        output[id] = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input) {
	if (input.empty()) return {};

	const size_t size = input.size();
	const size_t memory = size * sizeof(float);

	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputBuffer, outputBuffer;

	clGetPlatformIDs(1, &platform, nullptr);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
	queue = clCreateCommandQueue(context, device, 0, nullptr);
	program = clCreateProgramWithSource(context, 1, &geluKernelSource, nullptr, nullptr);
	const char* buildOptions = "-cl-fast-relaxed-math";
	clBuildProgram(program, 1, &device, buildOptions, nullptr, nullptr);
	kernel = clCreateKernel(program, "gelu", nullptr);

	inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, memory, (void*)input.data(), nullptr);
	outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, memory, nullptr, nullptr);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
	clSetKernelArg(kernel, 2, sizeof(int), &size);

	const size_t localSize = 256;
	const size_t globalSize = (size + localSize - 1) / localSize * localSize;

	clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);

	std::vector<float> output(size);
	clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, memory, output.data(), 0, nullptr, nullptr);

	clReleaseMemObject(inputBuffer);
	clReleaseMemObject(outputBuffer);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return output;
}