#include "gelu_ocl.h"
#include <CL/opencl.hpp>

struct Data
{
	cl::Context context;
	cl::CommandQueue queue;
	cl::Kernel kernel;
};
static Data data;

static const char* kernelCode = R"(__kernel void gelu(__global const float* input, __global float* result, int n)
{
  const int index = get_global_id(0);
  if (index < n) {
	const float x = input[index];
    const float tripleX = x * x * x;
    result[index] = 0.5f * x * (1.0f  + tanhf(0.7978f * (x + 0.044715f * tripleX)));
  }
})";

void initOpenClSession(int platformNum)
{
	using namespace cl;
	std::vector<cl::Platform> allPlatforms;
	cl::Platform::get(&allPlatforms);
	if (!allPlatforms.empty())
	{
		cl::Platform platform = allPlatforms[platformNum];
		std::vector<Device> allDevices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &allDevices);
		cl::Device mainDevice = allDevices.front();

		data.context = cl::Context({ mainDevice });

		data.queue = cl::CommandQueue(data.context, { mainDevice });

		cl::Program program(data.context, kernelCode);
		program.build({ mainDevice });

		data.kernel = cl::Kernel(program, "gelu");
	}
}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform)
{
	std::vector<float> output(input.size());
	const size_t mallocSize = input.size() * sizeof(float);

	initOpenClSession(platform);

	cl::Buffer inputBuffer(data.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mallocSize, const_cast<float*>(input.data()));
	cl::Buffer outputBuffer(data.context, CL_MEM_WRITE_ONLY, mallocSize);

	data.kernel.setArg(0, inputBuffer);
	data.kernel.setArg(1, outputBuffer);
	data.kernel.setArg(2, static_cast<int>(input.size()));

	data.queue.enqueueNDRangeKernel(data.kernel, cl::NullRange, cl::NDRange(input.size()), cl::NullRange);

	data.queue.finish();

	data.queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, mallocSize, output.data());
	return output;
}