#include "gelu_ocl.h"
#include <CL/opencl.hpp>
#include <vector>
#include <stdexcept>

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return std::vector<float>();
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found");
    }
    
    if (platform < 0 || platform >= platforms.size()) {
        throw std::runtime_error("Invalid platform index");
    }

    cl::Platform selected_platform = platforms[platform];
    std::vector<cl::Device> devices;
    selected_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    
    if (devices.empty()) {
        throw std::runtime_error("No GPU devices found on selected platform");
    }

    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    const std::string gelu_kernel_code = R"(
__kernel void gelu_activation(__global const float* input, 
                             __global float* output, 
                             const int count) {
    int idx = get_global_id(0);
    if (idx < count) {
        float x = input[idx];
        float tmp = 1.702f * x;
        float sigmoid = 1.0f / (1.0f + exp(-tmp));
        output[idx] = x * sigmoid;
    }
}
)";

    cl::Program program(context, gelu_kernel_code);
    if (program.build({device}) != CL_SUCCESS) {
        std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        throw std::runtime_error("OpenCL build failed: " + build_log);
    }

    size_t data_size = input.size();
    size_t buffer_size = data_size * sizeof(float);
    
    cl::Buffer input_buffer(context, CL_MEM_READ_ONLY, buffer_size);
    cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, buffer_size);

    queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, buffer_size, input.data());

    cl::Kernel kernel(program, "gelu_activation");
    kernel.setArg(0, input_buffer);
    kernel.setArg(1, output_buffer);
    kernel.setArg(2, static_cast<int>(data_size));

    size_t work_group_size = 256;
    size_t global_size = (data_size + work_group_size - 1) / work_group_size * work_group_size;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(work_group_size));

    std::vector<float> result(data_size);
    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, buffer_size, result.data());

    return result;
}