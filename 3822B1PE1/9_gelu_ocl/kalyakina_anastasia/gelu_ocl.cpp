#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <cmath>

const char* gelu_kernel = R"(
__kernel void gelu(__global const float* input, __global float* output, const int size) {
    const int idx = get_global_id(0);
    if (idx >= size) return;

    const float x = input[idx];
    float inner_value = 0.7978845608028654f * x * (1.0f + 0.044715f * x * x);
    float exp_value = exp(2.0f * inner_value);
    float tanh_value = (exp_value - 1.0f) / (exp_value + 1.0f);

    output[idx] = 0.5f * x * (1.0f + tanh_value);
}
)";

cl::Context context;
cl::CommandQueue queue;
cl::Kernel kernel;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform)
{
	std::vector<float> output(input.size());
	const size_t mallocSize = input.size() * sizeof(float);

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (!platforms.empty())
	{
		cl::Platform selected_platform = platforms[platform];
		std::vector<cl::Device> devices;

		selected_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		context = cl::Context(devices);

		queue = cl::CommandQueue(context, devices[0]);

		cl::Program program(context, gelu_kernel);
		program.build(devices);

		kernel = cl::Kernel(program, "gelu");
	}

	cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mallocSize, const_cast<float*>(input.data()));
	cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, mallocSize);

	kernel.setArg(0, input_buffer);
	kernel.setArg(1, output_buffer);
	kernel.setArg(2, static_cast<int>(input.size()));

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.size()), cl::NullRange);

	queue.finish();

	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, mallocSize, output.data());
	return output;
}