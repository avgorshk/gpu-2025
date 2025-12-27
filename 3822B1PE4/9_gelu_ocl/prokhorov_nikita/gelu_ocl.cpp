#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "gelu_ocl.h"
#include <iostream>
#include <vector>
#include <mutex>

static const char* gelu_kernel_source = R"(
__kernel void gelu(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    float x = input[idx];
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986856f;
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + 0.044715f * x3);
    float exp_val = exp(2.0f * inner);
    float tanh_val = (exp_val - 1.0f) / (exp_val + 1.0f);
    output[idx] = 0.5f * x * (1.0f + tanh_val);
}
)";

static cl::Context g_context;
static cl::CommandQueue g_queue;
static cl::Kernel g_kernel;
static bool g_initialized = false;
static std::mutex g_init_mutex;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform_id) {
	size_t n = input.size();
	if (n == 0) return {};

	std::lock_guard<std::mutex> lock(g_init_mutex);
	if (!g_initialized) {
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platform_id < 0 || platform_id >= static_cast<int>(platforms.size()))
			platform_id = 0;

		cl::Platform platform = platforms[platform_id];
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

		g_context = cl::Context(devices);
		g_queue = cl::CommandQueue(g_context, devices[0]);

		cl::Program program(g_context, gelu_kernel_source);
		program.build(devices);
		g_kernel = cl::Kernel(program, "gelu");
		g_initialized = true;
	}

	cl::Buffer input_buf(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		n * sizeof(float), const_cast<float*>(input.data()));
	cl::Buffer output_buf(g_context, CL_MEM_WRITE_ONLY, n * sizeof(float));

	g_kernel.setArg(0, input_buf);
	g_kernel.setArg(1, output_buf);
	g_kernel.setArg(2, static_cast<int>(n));

	g_queue.enqueueNDRangeKernel(g_kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);
	g_queue.finish();

	std::vector<float> result(n);
	g_queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, n * sizeof(float), result.data());
	return result;
}